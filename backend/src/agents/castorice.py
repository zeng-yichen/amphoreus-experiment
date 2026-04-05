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

import os
from google import genai
from google.genai import types
from backend.src.db import vortex as P


class Castorice:
    """
    Castorice: The Fact-Checker and Source Annotator.
    Reviews drafted posts against the client's local context files
    AND the live internet (via Google Search) to ensure factual accuracy,
    and independently traces every claim back to its source document.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get(
            "CASTORICE_GEMINI_MODEL", "gemini-2.5-flash"
        )
        self.client = genai.Client()

    def _get_local_context(self, company_keyword: str) -> str:
        """
        Load client context from transcripts, references, and content strategy.
        All three are needed for complete source annotation.
        """
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

        search_config = types.GenerateContentConfig(
            temperature=0.1,
            tools=[{"google_search": {}}],
        )

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
wording for factual corrections — preserve style, tone, and structure otherwise.
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

        last_err = None
        for attempt in range(4):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=system_instruction + "\n\n" + prompt,
                    config=search_config,
                )
                return self._parse_response(response.text)
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                # Avoid matching "rate" inside e.g. "generateContent" on 404 NOT_FOUND.
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
                    print(f"[Castorice] Retryable error (attempt {attempt + 1}/4), waiting {delay}s: {e}")
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

        # Extract [CORRECTED POST]
        if "[CORRECTED POST]" in raw:
            corrected_post = raw.split("[CORRECTED POST]", 1)[1].strip()

        return {
            "report": raw,
            "corrected_post": corrected_post or "",
            "annotated_post": annotated_post,
            "citation_comments": citation_comments,
        }
