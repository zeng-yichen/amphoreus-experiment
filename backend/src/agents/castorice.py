"""
Castorice — Fact-checker and source annotator.

Two tasks in one pass:
1. SOURCE ANNOTATION: independently traces every factual claim in the post
   back to source documents (transcripts, references, content strategy),
   producing an annotated version of the post with inline source comments.
2. FACT-CHECK: verifies claims against local context + Google Search,
   flags inaccuracies, and produces a corrected post if needed.

The annotated post is stored locally for editorial review (CE can see
sourcing without it touching the published text). Citation comments
are pushed as Ordinal post comments by Hyacinthia on confirmed push.
"""

import logging
import os
import re

import httpx
from google import genai
from google.genai import types

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defensive parsing helpers
# ---------------------------------------------------------------------------

# Patterns that indicate the LLM has started trailing content AFTER the
# [CORRECTED POST] body. If we hit one of these at the start of a line, we
# truncate the corrected post at that point.
#
# Note: these are operational heuristics for detecting model output drift,
# not learning-signal gates. Their purpose is to stop garbage from leaking
# into a published post if the LLM ignores the "stop after [CORRECTED POST]"
# instruction in the prompt.
_TRAILING_CONTAMINATION_LINE_PATTERNS = [
    # Bracketed section headers (the LLM recapping its own sections)
    re.compile(r"^\s*\[[A-Z][A-Z /\-]+\]\s*$"),
    # All-caps section labels with a trailing colon ("SUMMARY:", "NOTES:", etc.)
    re.compile(r"^\s*[A-Z][A-Z /\-]{3,}:\s*$"),
    # Horizontal rule separators that LinkedIn posts never contain
    re.compile(r"^\s*-{3,}\s*$"),
    re.compile(r"^\s*={3,}\s*$"),
    # Bold markdown section headers ("**SUMMARY:**", "**Corrections:**")
    re.compile(r"^\s*\*\*[A-Z][A-Za-z /\-]+:\*\*\s*$"),
    # Explicit end-of-report sentinels
    re.compile(r"^\s*\[END[A-Z /\-]*\]\s*$", re.IGNORECASE),
    # A line that looks like a restart of the Castorice report format
    re.compile(r"^\s*\[(ANNOTATED POST|CITATION COMMENTS|FACT-CHECK RESULTS|CORRECTED POST)\]\s*$"),
    # "SUMMARY:" as a standalone line
    re.compile(r"^\s*SUMMARY\s*:\s*$"),
]

# A URN is "placeholder" if its suffix is not a plain sequence of digits.
# Real LinkedIn/Ordinal organization/person URNs look like
# `urn:li:organization:11130470` — the suffix after the last colon is all
# digits. If the LLM invents `XXXXX`, `PLACEHOLDER`, `UNKNOWN`, or anything
# non-numeric, we strip the whole tag wrapper and leave the display name.
_TAG_WITH_URN_RE = re.compile(
    r"@\[([^\]]+)\]\(urn:li:(?:organization|person|member):([^\)]+)\)"
)


def _strip_trailing_contamination(text: str) -> str:
    """Truncate the corrected post at the first trailing section / recap marker.

    Walks the text line by line. Returns everything up to (but not
    including) the first line that matches a known contamination pattern.
    If no contamination is found, returns the input unchanged.
    """
    if not text:
        return text
    lines = text.split("\n")
    cut_at = None
    for i, line in enumerate(lines):
        for pat in _TRAILING_CONTAMINATION_LINE_PATTERNS:
            if pat.match(line):
                cut_at = i
                break
        if cut_at is not None:
            break
    if cut_at is None:
        return text
    # Trim trailing blank lines before the cut so we don't leave a dangling "\n\n"
    while cut_at > 0 and not lines[cut_at - 1].strip():
        cut_at -= 1
    trimmed = "\n".join(lines[:cut_at]).rstrip()
    if trimmed != text:
        logger.info(
            "[Castorice] trimmed trailing contamination at line %d "
            "(kept %d chars, discarded %d chars)",
            cut_at, len(trimmed), len(text) - len(trimmed),
        )
    return trimmed


def _strip_placeholder_urn_tags(text: str) -> str:
    """Replace any `@[Name](urn:li:*:NON_NUMERIC)` tag with just the bare name.

    Real URN suffixes are purely numeric. If the LLM slipped a placeholder
    (XXXXX, PLACEHOLDER, UNKNOWN, TBD, or any non-numeric value) into a tag
    wrapper, the LinkedIn publish will break. We defensively strip the tag
    syntax and leave the display name intact so the post is still valid,
    and log the occurrence so the operator can see the issue.
    """
    if not text or "urn:li:" not in text:
        return text

    stripped_count = 0
    def _replace(m: re.Match) -> str:
        nonlocal stripped_count
        display_name = m.group(1)
        urn_suffix = m.group(2).strip()
        # Real URN suffixes are digits only. Anything else is invented.
        if urn_suffix.isdigit():
            return m.group(0)  # keep the real tag
        stripped_count += 1
        return display_name

    new_text = _TAG_WITH_URN_RE.sub(_replace, text)
    if stripped_count:
        logger.warning(
            "[Castorice] Stripped %d placeholder-URN tag(s) from corrected post "
            "(LLM invented non-numeric URN suffixes). Flagged tags would have "
            "broken the published post.",
            stripped_count,
        )
    return new_text


class Castorice:
    """
    Castorice: The Fact-Checker and Source Annotator.
    Reviews drafted posts against the client's local context files
    AND the live internet (via Google Search) to ensure factual accuracy,
    and independently traces every claim back to its source document.

    Primary: Perplexity Sonar Pro (built-in web search, higher factuality).
    Fallback: Gemini with Google Search grounding.
    """

    def __init__(self, model_name: str | None = None):
        self._gemini_model = model_name or os.environ.get(
            "CASTORICE_GEMINI_MODEL", "gemini-2.5-flash"
        )
        self._gemini_client = genai.Client()
        self._perplexity_key = os.environ.get("PERPLEXITY_API_KEY", "")

    def _get_local_context(self, company_keyword: str) -> str:
        """
        Load client context from transcripts, references, and content strategy.
        All three are needed for complete source annotation.

        In Lineage mode (GCS + Supabase creds present and LINEAGE_COMPANY_ID
        set) we fetch transcripts and parallel research from Jacquard
        directly instead of reading the fly-local memory/ tree. The
        fly-local tree is empty in Lineage deployments — relying on it
        leaves Castorice fact-checking against nothing.

        Falls back to local-disk reads when not in Lineage mode (single-
        tenant amphoreus.app runs or tests).
        """
        # --- Lineage-mode path: pull from Jacquard's Supabase + GCS ---
        try:
            from backend.src.agents import lineage_fs_client as _lfs
            if _lfs.is_lineage_mode():
                lineage_ctx = self._get_lineage_context()
                if lineage_ctx:
                    return lineage_ctx
                # Empty in Lineage mode is still a valid answer — the user
                # may have no transcripts / research on file. Fall through
                # to return "" rather than silently loading stale fly-local.
                return ""
        except Exception as e:
            logger.warning("[Castorice] Lineage context fetch failed, falling back to local: %s", e)

        dirs_to_load = [
            ("transcripts", P.transcripts_dir(company_keyword)),
            ("references", P.references_dir(company_keyword)),
            ("content_strategy", P.content_strategy_dir(company_keyword)),
        ]
        context_parts: list[str] = []

        for label, directory in dirs_to_load:
            if not directory.exists():
                continue
            for filepath in sorted(directory.rglob("*")):
                if filepath.is_file() and filepath.suffix.lower() in (".txt", ".md"):
                    try:
                        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
                        if text:
                            context_parts.append(
                                f"\n--- SOURCE [{label}]: {filepath.name} ---\n{text}\n"
                            )
                    except Exception as e:
                        print(f"[Castorice] Failed to read {filepath.name}: {e}")

        return "".join(context_parts)

    def _get_lineage_context(self) -> str:
        """
        Assemble Castorice's local-context block from Jacquard data.

        Emits the same ``--- SOURCE [label]: filename ---`` framing as the
        fly-local reader so the downstream system prompt doesn't need to
        know which backend produced it.

        Sources:
          - transcripts/ — ``fetch_meeting_transcripts`` (Supabase meetings
            + GCS-hosted transcript bodies)
          - research/    — ``fetch_parallel_research`` (company + person
            research outputs)
          - context/     — ``fetch_context_files`` (uploaded brand docs +
            synthesized account.md)

        Requires LINEAGE_COMPANY_ID. If LINEAGE_USER_SLUG is present we
        resolve to a specific FOC user and pull their meetings/research;
        otherwise we return just the company-level context files.
        """
        import os as _os
        from backend.src.agents import jacquard_direct as _jd

        company_id = _os.environ.get("LINEAGE_COMPANY_ID", "").strip()
        if not company_id:
            return ""

        user_slug = _os.environ.get("LINEAGE_USER_SLUG", "").strip() or None
        user: dict | None = None
        if user_slug:
            try:
                user = _jd.resolve_user_by_slug(company_id, user_slug)
            except Exception as e:
                logger.warning(
                    "[Castorice] resolve_user_by_slug(%s, %s) failed: %s",
                    company_id, user_slug, e,
                )

        context_parts: list[str] = []

        # Transcripts (user-scoped). Skip cleanly if we have no user.
        if user:
            try:
                transcripts = _jd.fetch_meeting_transcripts(
                    user.get("id", ""),
                    user.get("email"),
                    bool(user.get("is_internal")),
                )
                for item in transcripts:
                    text = (item.get("content") or "").strip()
                    if text:
                        context_parts.append(
                            f"\n--- SOURCE [transcripts]: {item.get('filename', 'transcript.md')} ---\n{text}\n"
                        )
            except Exception as e:
                logger.warning("[Castorice] fetch_meeting_transcripts failed: %s", e)

            try:
                research = _jd.fetch_parallel_research(company_id, user.get("id", ""))
                for item in research:
                    text = (item.get("content") or "").strip()
                    if text:
                        context_parts.append(
                            f"\n--- SOURCE [research]: {item.get('filename', 'research.md')} ---\n{text}\n"
                        )
            except Exception as e:
                logger.warning("[Castorice] fetch_parallel_research failed: %s", e)

        # Context files (company-scoped; account.md needs user for richness).
        try:
            ctx_files = _jd.fetch_context_files(company_id, user)
            for item in ctx_files:
                text = (item.get("content") or "").strip()
                if text:
                    context_parts.append(
                        f"\n--- SOURCE [context]: {item.get('filename', 'context.md')} ---\n{text}\n"
                    )
        except Exception as e:
            logger.warning("[Castorice] fetch_context_files failed: %s", e)

        return "".join(context_parts)

    def fact_check_post(
        self,
        company_keyword: str,
        post_content: str,
    ) -> dict:
        """
        Annotate and fact-check a post.

        Args:
            company_keyword: Client identifier used to locate source files.
            post_content: The clean post text produced by Stelle.

        Returns:
            dict with keys:
                "report"             — full fact-check text (str)
                "corrected_post"     — post text after factual fixes (str)
                "annotated_post"     — post text with inline source comments (str)
                "citation_comments"  — list of formatted comment strings for
                                       Ordinal post comments (list[str])
        """
        local_context = self._get_local_context(company_keyword)

        system_instruction = """\
You are Castorice, a ruthless fact-checker and meticulous source annotator for \
a LinkedIn ghostwriting agency. You work on clean post text that has no \
embedded citations — your job is to independently trace every factual claim \
back to its source and verify it.

You have TWO sources of truth:
1. LOCAL CONTEXT: Transcripts, references, and content strategy documents \
   provided below. These are the ground-truth documents the post should be \
   grounded in.
2. THE WEB: Use Google Search to verify any external claims, statistics, \
   historical events, or industry trends.

IMPORTANT — SOURCE CITATION REQUIREMENTS:
Two types of claims require inline source comments in the annotated post:
1. EXTERNAL FACTS/FIGURES: Any specific fact, statistic, or figure relating to \
   public data (industry trends, third-party benchmarks, company funding, etc.) \
   MUST have an inline comment citing the source URL or publication.
2. CLIENT/COMPANY-SPECIFIC INFO: Any specific claim about the client or their \
   company (product details, internal metrics, team decisions, customer wins, etc.) \
   MUST have an inline comment citing the exact supporting quote from a call \
   transcript. If no transcript quote supports the claim, flag it explicitly as \
   UNSOURCED CLIENT CLAIM in the [FACT-CHECK RESULTS] section.

IMPORTANT — QUASI-FICTIONAL ANECDOTES:
LinkedIn thought leadership often packages real insights inside illustrative \
anecdotes ("I was sitting across from a CTO who told me..."). These are \
storytelling devices, not factual claims. Let them through as long as:
  (a) the scenario is plausible given the client's role, industry, and expertise, AND
  (b) the underlying business insight or lesson is accurate.
Only flag an anecdote if it contains a specific, checkable claim that is \
demonstrably false.

CRITICAL — METRIC PRECISION TRAPS:
Catch all of the following:
1. COMPONENT vs. END-TO-END METRICS: Verify that cited numbers measure the \
   full pipeline, not just one component.
2. SUPERLATIVES AND UNIQUENESS CLAIMS: "first ever", "no one else", \
   "unprecedented" require proof. Search for competitors before approving.
3. BENCHMARK SCOPE: "beat every competitor" requires verifying the exact \
   comparison methodology.
4. METRIC MISATTRIBUTION: Revenue vs. ARR, users vs. DAU, funding vs. \
   valuation, TTFA vs. round-trip latency.
5. TEMPORAL ACCURACY: Flag stats where a more recent figure exists publicly.
6. COPY-PASTE QUALITY: Check for (a) typos; (b) accidental run-on URLs caused by a \
missing space after a period — these look like "teams.I could" or "use.No wonder", \
where the next word runs directly into the URL/period without a space; and (c) proper \
nouns (companies, people) that should be tagged but are not — correctly tagged proper \
nouns look like `@[OpenAI](urn:li:organization:11130470)` or `@[Julien Chaumond]`. \
Flag every company name and person's name in the post that appears untagged.

CRITICAL — NEVER INVENT URNs:
Do NOT invent, fabricate, placeholder, or guess Ordinal URNs. If a company or \
person name in the post is not already tagged and you don't have the real \
`urn:li:organization:<numeric_id>` or `urn:li:person:<numeric_id>` from the \
original draft, DO NOT wrap the name in tag syntax in the [CORRECTED POST]. \
Leave it as plain text exactly as it was. Flag the missing tag in \
[CORRECTIONS NEEDED] for human action (e.g. "Add LinkedIn tag for ADC Therapeutics"). \
Placeholder URNs like `urn:li:organization:XXXXX`, `:PLACEHOLDER`, `:UNKNOWN`, \
`:TBD`, or any non-numeric stand-in are forbidden and will break the published \
post. When in doubt, keep the plain text.

---

YOUR OUTPUT FORMAT (in this exact order):

[ANNOTATED POST]
The full post text, with one source comment appended after each paragraph \
that contains a traceable factual claim. Use this format for each comment:
<!-- source: {filename} | "{exact quote or paraphrase from source}" -->
If a paragraph contains no traceable factual claim (framing, thesis, inference), \
do not add a comment. If a claim traces to the web rather than local context, \
use: <!-- source: web | "{source URL or publication}" -->

[CITATION COMMENTS]
For each annotated claim, output one entry formatted as:
Claim: "<the exact sentence or phrase containing the factual claim>"
Source: <filename or URL> | "<supporting quote>"

Separate entries with a blank line. These will be posted verbatim as \
Ordinal post comments for editorial review.

[FACT-CHECK RESULTS]
For each checkable claim:
"quoted claim" — ✅/⚠️/❌/ℹ️ Verdict
Explanation with source URLs where applicable.

[INTERNAL VERIFICATION]: (Pass/Fail) Does the post align with transcripts?
[EXTERNAL VERIFICATION]: (Pass/Fail) Do industry claims hold up to web search?
[COPY QUALITY]: (Pass/Fail) Any typos, accidental run-on URLs (missing space after \
period), or untagged proper nouns? List each issue found.
[CORRECTIONS NEEDED]: List specific fixes. If none, say "None."

[CORRECTED POST]
The full LinkedIn post text: if corrections were needed, the complete updated \
draft; if none were needed, repeat the original draft verbatim. Only change \
wording for factual corrections — preserve style, tone, and structure otherwise. \
Preserve existing `@[Name](urn:li:organization:NNN)` tags verbatim. Do NOT \
introduce new tags unless you have the real numeric URN from the original draft.

STOP AFTER [CORRECTED POST]. Do not add any trailing summary, recap, SUMMARY \
section, closing notes, section headers, horizontal rules, or any other \
content after the corrected post body. The corrected post ends at the last \
line of the LinkedIn post text and nothing follows. Everything after the \
post's final line will be treated as contamination and discarded.
"""

        prompt = f"""\
<local_context>
{local_context}
</local_context>

<drafted_post>
{post_content}
</drafted_post>

Produce your annotation and fact-check in the required format.
"""

        import time as _time

        # --- Primary: Perplexity Sonar Pro (built-in web search) ---
        if self._perplexity_key:
            try:
                return self._fact_check_perplexity(system_instruction, prompt, post_content)
            except Exception as pplx_err:
                logger.warning("[Castorice] Perplexity failed: %s — falling back to Gemini", pplx_err)

        # --- Fallback: Gemini with Google Search grounding ---
        search_config = types.GenerateContentConfig(
            temperature=0.1,
            tools=[{"google_search": {}}],
        )

        last_err = None
        for attempt in range(4):
            try:
                response = self._gemini_client.models.generate_content(
                    model=self._gemini_model,
                    contents=system_instruction + "\n\n" + prompt,
                    config=search_config,
                )
                return self._parse_response(response.text)
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if any(
                    s in err_str
                    for s in (
                        "429",
                        "503",
                        "rate_limit",
                        "resource exhausted",
                        "overloaded",
                        "timeout",
                    )
                ):
                    delay = min(20.0, 2.0 * (2 ** attempt))
                    print(f"[Castorice] Gemini retryable error (attempt {attempt + 1}/4), waiting {delay}s: {e}")
                    _time.sleep(delay)
                    continue
                return {
                    "report": f"Fact-check failed due to API error: {e}",
                    "corrected_post": post_content,
                    "annotated_post": "",
                    "citation_comments": [],
                }

        return {
            "report": f"Fact-check failed after retries: {last_err}",
            "corrected_post": post_content,
            "annotated_post": "",
            "citation_comments": [],
        }

    def _fact_check_perplexity(
        self, system_instruction: str, prompt: str, post_content: str,
    ) -> dict:
        """Fallback fact-checker using Perplexity Sonar (has built-in web search)."""
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {self._perplexity_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar-pro",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 4000,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        body = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        if not body:
            return {
                "report": "Perplexity returned empty response",
                "corrected_post": post_content,
                "annotated_post": "",
                "citation_comments": [],
            }
        logger.info("[Castorice] Perplexity fallback succeeded")
        return self._parse_response(body)

    def _parse_response(self, raw: str) -> dict:
        """Parse the model output into structured fields."""
        annotated_post = ""
        corrected_post = ""
        citation_comments: list[str] = []

        # Extract [ANNOTATED POST] — everything up to the next section tag
        if "[ANNOTATED POST]" in raw:
            after = raw.split("[ANNOTATED POST]", 1)[1]
            for next_tag in ("[CITATION COMMENTS]", "[FACT-CHECK RESULTS]", "[CORRECTED POST]"):
                if next_tag in after:
                    annotated_post = after.split(next_tag, 1)[0].strip()
                    break
            else:
                annotated_post = after.strip()

        # Extract [CITATION COMMENTS] — collect blank-line-separated entries
        if "[CITATION COMMENTS]" in raw:
            after = raw.split("[CITATION COMMENTS]", 1)[1]
            for next_tag in ("[FACT-CHECK RESULTS]", "[CORRECTED POST]"):
                if next_tag in after:
                    cite_block = after.split(next_tag, 1)[0].strip()
                    break
            else:
                cite_block = after.strip()
            citation_comments = [e.strip() for e in cite_block.split("\n\n") if e.strip()]

        # Extract [CORRECTED POST] and defensively trim trailing contamination.
        #
        # The prompt tells the LLM to stop after the corrected post body, but
        # models occasionally append a SUMMARY / recap / closing metadata
        # block. Naive `split("[CORRECTED POST]", 1)[1]` would bundle all of
        # that into the post text. We trim at the first trailing marker that
        # cannot be part of a normal LinkedIn post body, and also strip any
        # placeholder-URN tags (the other known failure mode where the LLM
        # invents `urn:li:organization:XXXXX` or similar).
        if "[CORRECTED POST]" in raw:
            tail = raw.split("[CORRECTED POST]", 1)[1].strip()
            corrected_post = _strip_trailing_contamination(tail)
            corrected_post = _strip_placeholder_urn_tags(corrected_post)

        return {
            "report": raw,
            "corrected_post": corrected_post or "",
            "annotated_post": annotated_post,
            "citation_comments": citation_comments,
        }
