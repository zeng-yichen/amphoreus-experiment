"""Aglaea — the client-comfort critic.

A second critic alongside Irontomb. Where Irontomb asks *"will a LinkedIn
reader stop scrolling?"* (engagement), Aglaea asks *"will this FOC user
actually publish this draft as-is?"* (comfort / voice fidelity). A post
that ships through Amphoreus must pass both — a viral-sounding draft
the client would never publish is worthless, and a perfectly-on-voice
draft that no one engages with is also worthless.

Contract:

  evaluate_client_comfort(draft_text, user_id=None, user_slug=None, company_slug=None)
      → {
          "score":          0..10,
          "flagged_spans":  [{"quote": str, "reason": str, "suggestion": str}, ...],
          "summary":        str,    # one-sentence takeaway
          "_draft_hash":    str,
          "_cost_usd":      float,
          "_model":         str,
        }

Signal:
  * Last 6 months of the user's LinkedIn posts (voice reference).
  * ``draft_feedback`` rows on this user's past drafts (explicit
    operator / Ordinal / client comments flagging prior drafts).
  * Past (pre_revision_content, published_text) deltas from
    ``local_posts`` — the implicit "what a human softened before
    shipping" signal.

Never raises. Failures return a stub ``{"_error": ...}`` so Stelle's
inner loop can keep iterating rather than crashing.

This is the v0 — one LLM call, no tools, heuristics-only signal
aggregation. Once ``draft_feedback`` + edit-history corpora grow, we
can fine-tune or prompt-calibrate this more precisely per client.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import anthropic

logger = logging.getLogger(__name__)

_AGLAEA_MODEL = "claude-sonnet-4-5"
_AGLAEA_MAX_TOKENS = 2000
# Rolling window for voice reference. Matches the 6-month linkedin_posts
# mirror trim — older posts have drifted stylistically.
_VOICE_LOOKBACK_DAYS = 180
# Number of voice-reference posts to include in the prompt. More = better
# signal but more tokens. 20 is enough to see pattern without bloating.
_VOICE_REFERENCE_COUNT = 20
# Cap on each voice-reference post text length (chars). Long posts get
# truncated to keep the prompt tight.
_VOICE_POST_CHAR_CAP = 1200
# Max feedback rows to pull (most recent). Feedback is usually short; 20
# is plenty without drowning the prompt.
_MAX_FEEDBACK_ROWS = 20
# Max edit-delta pairs. Each pair eats ~2-3k chars; 5 is the sweet spot.
_MAX_EDIT_DELTAS = 5
# Transcript substrate (added 2026-04-30). Pull recent client/content
# interviews — the unscripted-voice ground truth.
_TRANSCRIPT_LOOKBACK_DAYS = 90
_MAX_TRANSCRIPTS = 5
_TRANSCRIPT_CHAR_CAP = 6000  # per-transcript char cap; full text in Supabase if needed

# Process-local cache so Stelle's iterative calls on the same draft don't
# re-query + re-embed each time. Keyed on (draft_hash, user_id).
_CACHE_TTL_SECONDS = 3600
_cache: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}


def _draft_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def evaluate_client_comfort(
    draft_text: str,
    *,
    user_id: Optional[str] = None,
    user_slug: Optional[str] = None,
    company_slug: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate how likely the target FOC user is to publish ``draft_text`` as-is.

    Identity resolution: pass one of ``user_id`` (preferred — UUID direct
    lookup), ``user_slug`` (Jacquard ``users.slug``), or fall back to
    ``company_slug`` (evaluator then operates company-wide, weaker signal).

    Returns a dict with keys ``score``, ``flagged_spans``, ``summary``,
    and ``_draft_hash`` / ``_cost_usd`` / ``_model`` bookkeeping.
    Failures return ``{"_error": ..., "score": None, "flagged_spans": []}``
    so callers can degrade gracefully.
    """
    if not draft_text or not draft_text.strip():
        return {"_error": "draft_text is empty", "score": None, "flagged_spans": []}

    resolved_user_id = _resolve_user_id(user_id, user_slug, company_slug)
    cache_key = (_draft_hash(draft_text), resolved_user_id or "")
    now = time.time()
    cached = _cache.get(cache_key)
    if cached is not None:
        value, expires = cached
        if expires > now:
            return value

    try:
        signal = _build_voice_signal(
            user_id=resolved_user_id,
            company_slug=company_slug,
            user_slug=user_slug,
        )
    except Exception as exc:
        logger.warning("[Aglaea] signal build failed: %s", exc)
        signal = {
            "voice_posts": [],
            "feedback": [],
            "edit_deltas": [],
            "user_label": user_slug or company_slug or "unknown",
        }

    try:
        result = _call_model(draft_text, signal)
    except Exception as exc:
        logger.exception("[Aglaea] model call failed")
        return {
            "_error": f"model call failed: {str(exc)[:200]}",
            "score": None,
            "flagged_spans": [],
            "_draft_hash": _draft_hash(draft_text),
        }

    result["_draft_hash"] = _draft_hash(draft_text)
    _cache[cache_key] = (result, now + _CACHE_TTL_SECONDS)
    return result


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------

def _resolve_user_id(
    user_id: Optional[str],
    user_slug: Optional[str],
    company_slug: Optional[str],
) -> Optional[str]:
    """Turn whatever identity hints the caller has into a ``users.id`` UUID."""
    if user_id:
        return user_id.strip()
    if not user_slug and not company_slug:
        return None
    try:
        from backend.src.lib.company_resolver import resolve_to_company_and_user
        # Try user_slug first if available; fall back to company_slug.
        _co, _u = resolve_to_company_and_user(user_slug or company_slug)
        return _u
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Signal aggregation
# ---------------------------------------------------------------------------

def _build_voice_signal(
    *,
    user_id: Optional[str],
    company_slug: Optional[str],
    user_slug: Optional[str],
) -> dict[str, Any]:
    """Pull the three signal sources into a single dict for the prompt."""
    from backend.src.db.amphoreus_supabase import _get_client
    sb = _get_client()
    if sb is None:
        return {"voice_posts": [], "feedback": [], "edit_deltas": [], "user_label": user_slug or company_slug or "unknown"}

    user_label = user_slug or company_slug or (user_id or "unknown")
    user_email: Optional[str] = None
    creator_username: Optional[str] = None

    # Resolve user_id → (email, linkedin_username) so we can query the
    # user's posts + feedback + drafts.
    if user_id:
        try:
            rows = (
                sb.table("users")
                  .select("email, first_name, last_name, linkedin_url")
                  .eq("id", user_id)
                  .limit(1)
                  .execute()
                  .data
                or []
            )
            if rows:
                row = rows[0]
                user_email = (row.get("email") or "").strip().lower() or None
                first = (row.get("first_name") or "").strip()
                last = (row.get("last_name") or "").strip()
                if first or last:
                    user_label = f"{first} {last}".strip()
                url = (row.get("linkedin_url") or "").strip()
                m = re.search(r"linkedin\.com/in/([^/?#]+)", url)
                if m:
                    creator_username = m.group(1).strip().lower().rstrip("/")
        except Exception as exc:
            logger.debug("[Aglaea] user lookup failed: %s", exc)

    voice_posts = _fetch_voice_reference_posts(sb, creator_username)
    feedback = _fetch_past_feedback(sb, user_id)
    edit_deltas = _fetch_edit_deltas(sb, user_id, user_email)
    # 2026-04-30: transcripts added to Aglaea's substrate. The user's
    # client-interview transcripts are the strongest unscripted-voice
    # signal we have — strictly more signal than LinkedIn posts (which
    # are already polished) for predicting "would this user publish
    # this as-is?" Previously Aglaea didn't see transcripts at all.
    transcripts = _fetch_recent_transcripts(company_slug, user_id)

    # Post bundle — per-post joined view (body + engagement + comments
    # + delta). Complements the flat aggregate lists above: aggregate
    # views give wide voice/pattern calibration, the bundle gives
    # literal "this draft got this comment" grounding. Published,
    # InFlight, and Rejected drafts are all surfaced; the Rejected
    # class is Aglaea's most important signal — drafts the client
    # explicitly refused, with the paired comments that did the
    # refusing.
    post_bundle = ""
    if company_slug:
        try:
            from backend.src.services.post_bundle import build_post_bundle
            # Per-FOC scope: pass user_id so at multi-FOC companies
            # (Virio/Trimble/Commenda) Aglaea sees only this user's
            # drafts + their comments, not every sibling FOC's. Stops
            # cross-FOC bleed AND keeps bundle size bounded
            # (2026-04-23 ARG_MAX incident).
            post_bundle = build_post_bundle(company_slug, user_id=user_id)
        except Exception as exc:
            logger.debug("[Aglaea] post bundle build failed: %s", exc)

    return {
        "user_label":    user_label,
        "voice_posts":   voice_posts,
        "feedback":      feedback,
        "edit_deltas":   edit_deltas,
        "transcripts":   transcripts,
        "post_bundle":   post_bundle,
    }


def _fetch_voice_reference_posts(sb, creator_username: Optional[str]) -> list[dict[str, Any]]:
    """The user's last N LinkedIn posts — voice / register reference."""
    if not creator_username:
        return []
    since_iso = (
        datetime.now(timezone.utc) - timedelta(days=_VOICE_LOOKBACK_DAYS)
    ).isoformat()
    try:
        rows = (
            sb.table("linkedin_posts")
              .select("post_text, posted_at, total_reactions, total_comments, hook")
              .eq("creator_username", creator_username)
              .gte("posted_at", since_iso)
              .order("posted_at", desc=True)
              .limit(_VOICE_REFERENCE_COUNT)
              .execute()
              .data
            or []
        )
    except Exception as exc:
        logger.debug("[Aglaea] voice posts fetch failed: %s", exc)
        return []
    out = []
    for r in rows:
        txt = (r.get("post_text") or "").strip()
        if not txt:
            continue
        out.append({
            "posted_at": r.get("posted_at"),
            "reactions": r.get("total_reactions") or 0,
            "comments":  r.get("total_comments") or 0,
            "text":      txt[:_VOICE_POST_CHAR_CAP],
        })
    return out


#: Body prefixes we skip when building Aglaea's feedback signal. These
#: are Amphoreus-generated internal rationales (``Hyacinthia._save_draft_map_entry``
#: writes them to Ordinal as the first ``/posts/{id}/comments`` entry)
#: that contaminate the "don't say this" signal Aglaea is trying to
#: surface. Matching by prefix — the body shape is stable.
_FEEDBACK_SKIP_PREFIXES: tuple[str, ...] = (
    # Hyacinthia's why_post auto-comment at push time (company
    # requirement to keep why_post; we filter the feedback-stream
    # echo so Aglaea doesn't read our own rationale as client input).
    "why we're posting this (internal):",
    # Legacy Castorice fact-check header (some older format).
    "## castorice fact-check",
    # Castorice citation_comments — Hyacinthia auto-posts each
    # citation as its own Ordinal comment at push time. Format from
    # castorice.py:384 — ``Claim: "<sentence>"\nSource: <ref>``. These
    # are our own fact-check receipts, not reviewer input, so filter.
    'claim: "',
)


def _fetch_past_feedback(sb, user_id: Optional[str]) -> list[dict[str, Any]]:
    """Past ``draft_feedback`` rows for THIS user's drafts.

    Joins ``draft_feedback.draft_id → local_posts.user_id``. Historic
    comments teach Aglaea what the client / operator has explicitly
    flagged before — the literal "don't say this" signal.

    Filters out Amphoreus-generated rationale rows (``Why we're posting
    this (internal):`` that Hyacinthia auto-posts to Ordinal at push
    time) — those are our own commentary, not client signal, and would
    bias the evaluator toward our priors.
    """
    if not user_id:
        return []
    try:
        draft_rows = (
            sb.table("local_posts")
              .select("id")
              .eq("user_id", user_id)
              .limit(500)
              .execute()
              .data
            or []
        )
        draft_ids = [r["id"] for r in draft_rows if r.get("id")]
        if not draft_ids:
            return []
        fb_rows = (
            sb.table("draft_feedback")
              .select("body, selected_text, source, author_email, created_at")
              .in_("draft_id", draft_ids)
              .order("created_at", desc=True)
              .limit(_MAX_FEEDBACK_ROWS * 3)  # over-pull — we filter below
              .execute()
              .data
            or []
        )
    except Exception as exc:
        logger.debug("[Aglaea] feedback fetch failed: %s", exc)
        return []
    out = []
    for r in fb_rows:
        body = (r.get("body") or "").strip()
        if not body:
            continue
        body_lc = body.lower()
        if any(body_lc.startswith(p) for p in _FEEDBACK_SKIP_PREFIXES):
            continue
        out.append({
            "source":        r.get("source"),
            "author_email":  r.get("author_email"),
            "selected_text": (r.get("selected_text") or "")[:200],
            "body":          body[:600],
        })
        if len(out) >= _MAX_FEEDBACK_ROWS:
            break
    return out


def _fetch_edit_deltas(
    sb,
    user_id: Optional[str],
    user_email: Optional[str],
) -> list[dict[str, Any]]:
    """Past (pre_revision, published) pairs from this user's drafts.

    The implicit "a human softened this" signal. We return up to
    ``_MAX_EDIT_DELTAS`` of the most recent drafts where both
    ``pre_revision_content`` and ``content`` are set and differ.
    """
    if not user_id:
        return []
    try:
        rows = (
            sb.table("local_posts")
              .select("content, pre_revision_content, created_at")
              .eq("user_id", user_id)
              .not_.is_("pre_revision_content", "null")
              .order("created_at", desc=True)
              .limit(_MAX_EDIT_DELTAS * 3)  # over-pull then filter for real deltas
              .execute()
              .data
            or []
        )
    except Exception as exc:
        logger.debug("[Aglaea] edit deltas fetch failed: %s", exc)
        return []
    out = []
    for r in rows:
        pre = (r.get("pre_revision_content") or "").strip()
        post = (r.get("content") or "").strip()
        if not pre or not post or pre == post:
            continue
        out.append({
            "pre":  pre[:1500],
            "post": post[:1500],
        })
        if len(out) >= _MAX_EDIT_DELTAS:
            break
    return out


def _fetch_recent_transcripts(
    company_slug: Optional[str],
    user_id: Optional[str],
) -> list[dict[str, Any]]:
    """Pull recent client/content interviews for voice grounding.

    Reads from the same canonical Supabase helper Cyrene + Tribbie use,
    so per-FOC scoping is handled identically across agents. Returns at
    most ``_MAX_TRANSCRIPTS`` rows, most recent first; empty list on any
    failure (Aglaea degrades gracefully — voice_posts + edit_deltas
    still cover the basics).
    """
    if not company_slug:
        return []
    try:
        from backend.src.db.amphoreus_supabase import get_client_transcripts
        # NOTE: get_client_transcripts is company-scoped (no user_id
        # filter). At multi-FOC clients (Trimble heather/mark, Commenda
        # logan/sam) Aglaea will see both FOCs' transcripts. For voice
        # grounding this is acceptable — the model can pattern-match
        # against the relevant speaker via the natural register
        # differences. If false-positive cross-FOC bleed shows up in
        # practice, narrow this later.
        rows = get_client_transcripts(
            company_slug, limit=_MAX_TRANSCRIPTS * 2,
        )
    except Exception as exc:
        logger.debug("[Aglaea] transcripts fetch failed: %s", exc)
        return []

    if not rows:
        return []

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=_TRANSCRIPT_LOOKBACK_DAYS)
    ).isoformat()
    out: list[dict[str, Any]] = []
    for r in rows:
        posted_at = (r.get("posted_at") or "")
        if posted_at and posted_at < cutoff:
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "filename":  r.get("filename") or "(untitled)",
            "posted_at": posted_at or "",
            "text":      text,
        })
        if len(out) >= _MAX_TRANSCRIPTS:
            break
    return out


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are Aglaea — a client-comfort critic for LinkedIn drafts.

Your job is NOT to evaluate whether the draft is good or will land with
readers. Irontomb does that. Your job is ONE question:

  "Would THIS specific FOC user publish THIS draft as-is?"

A draft fails your bar when the user would:
  * refuse to post it ("this isn't how I talk"),
  * heavily edit it before shipping (softening, removing claims, cutting
    jargon they don't use),
  * request a total rewrite.

A draft passes when it reads like something the user has already posted
or would comfortably post without edits.

Your signal comes from FOUR sources the caller provides. Use them all:

  TRANSCRIPTS: client/content interviews — the user speaking unscripted.
               STRONGEST voice signal. Their LinkedIn posts have already
               been polished by Stelle + their own edits; the transcript
               captures their natural cadence, vocabulary, and register
               BEFORE polish. When the draft's tone diverges from the
               transcript voice, that's the strongest "they wouldn't
               publish this" signal you have.

  VOICE_POSTS: their recent LinkedIn posts (≈6 months). Polished voice —
               useful for sentence rhythm, opener/closer patterns,
               post-shape conventions. If the draft uses words/frames
               absent from every voice post, that's a flag.

  FEEDBACK:    past comments on their prior drafts (operator + client).
               Explicit flags — the literal "don't say this" signal.

  EDIT_DELTAS: past (draft, published) pairs showing what a human
               softened before shipping. Implicit flags. Repeated edit
               patterns predict future edits.

Be specific. Cite quotes from the draft AND from the substrate (e.g.,
"this line uses 'leverage' but every TRANSCRIPTS sample uses plain
verbs — they don't talk like that"). Vague verdicts that don't point
at substrate are not useful — the operator can't act on them.

Be conservative on the score —
6 means "needs real edits," 8 means "minor softening," 10 means "ship
it unchanged."

Output ONLY a JSON object with this exact shape — no prose, no markdown,
no explanation outside the JSON:

  {
    "score": <int 0-10>,
    "summary": "<one sentence takeaway, ≤120 chars>",
    "flagged_spans": [
      {
        "quote":      "<verbatim substring of the draft, ≤200 chars>",
        "reason":     "<why this line risks an edit, ≤200 chars>",
        "suggestion": "<concrete rewrite, ≤200 chars>",
        "substrate_evidence": "<verbatim 3-30 word quote from TRANSCRIPTS / VOICE_POSTS / FEEDBACK / EDIT_DELTAS that grounds the flag, OR empty string if the flag is purely structural>"
      }
    ]
  }

If the draft is on-voice enough to ship, ``flagged_spans`` MAY be empty.

The ``substrate_evidence`` field is REQUIRED on every flag. Empty
string is allowed ONLY when the flag is structural (e.g., "the closer
doesn't connect to the opener" — no substrate needed). Any flag that
generalizes about "the user wouldn't say this" / "this isn't their
voice" / "they'd soften this" MUST cite a concrete substrate quote
showing how the user actually talks. A flag without grounding is a
prior, not a signal — the operator can't act on it.
"""


def _render_signal_block(signal: dict[str, Any]) -> str:
    """Format the three signal sources into the user message."""
    parts: list[str] = []
    parts.append(f"TARGET FOC USER: {signal.get('user_label') or 'unknown'}")
    parts.append("")

    vp = signal.get("voice_posts") or []
    if vp:
        parts.append(f"=== VOICE_POSTS ({len(vp)} recent LinkedIn posts, most-recent first) ===")
        for i, p in enumerate(vp, 1):
            parts.append(
                f"\n[{i}] posted={p.get('posted_at')}  reactions={p.get('reactions')}  comments={p.get('comments')}"
            )
            parts.append(p.get("text") or "")
    else:
        parts.append("=== VOICE_POSTS ===")
        parts.append("(no recent posts — operate on draft fit to topic + feedback signal)")
    parts.append("")

    fb = signal.get("feedback") or []
    if fb:
        parts.append(f"=== FEEDBACK ({len(fb)} past comments — highest-weight signal) ===")
        parts.append(
            "Comments from the CLIENT themselves (author_email matching the "
            "user's domain) carry the most weight — they are literal "
            "'don't publish this' votes. Weight operator (@virio.ai) "
            "comments less."
        )
        for i, f in enumerate(fb, 1):
            sel = f.get("selected_text") or ""
            who = f.get("author_email") or "unknown"
            sel_bit = f'  on: "{sel}"' if sel else ""
            parts.append(f"\n[{i}] by {who}  source={f.get('source')}{sel_bit}")
            parts.append(f.get("body") or "")
    else:
        parts.append("=== FEEDBACK ===\n(no past feedback for this user)")
    parts.append("")

    ed = signal.get("edit_deltas") or []
    if ed:
        parts.append(f"=== EDIT_DELTAS ({len(ed)} pre→post pairs) ===")
        for i, d in enumerate(ed, 1):
            parts.append(f"\n[{i}] PRE (Stelle draft):\n{d.get('pre') or ''}")
            parts.append(f"\n[{i}] POST (what actually shipped):\n{d.get('post') or ''}")
    else:
        parts.append("=== EDIT_DELTAS ===\n(no past draft-vs-published pairs for this user)")
    parts.append("")

    # 2026-04-30 added — transcripts are the unscripted-voice signal.
    # LinkedIn posts are POLISHED voice (already through Stelle + the
    # client's own edits); transcripts are how the client actually
    # speaks. Strongest single signal for the "would they publish this
    # as-is?" question, because Aglaea can directly compare the draft's
    # cadence/word choice/register to the user's natural cadence.
    tr = signal.get("transcripts") or []
    if tr:
        parts.append(f"=== TRANSCRIPTS ({len(tr)} client interview transcripts, most-recent first) ===")
        parts.append(
            "These are the user speaking unscripted in client/content "
            "interviews. STRONGEST voice signal — pattern-match against "
            "this when judging draft cadence, vocabulary, sentence "
            "rhythm, and stance formality."
        )
        for i, t in enumerate(tr, 1):
            parts.append(
                f"\n[{i}] {t.get('filename', '(untitled)')}  "
                f"posted_at={t.get('posted_at', '?')[:10]}  "
                f"chars={len(t.get('text') or '')}"
            )
            # Truncate per-transcript to keep prompt bounded — the full
            # corpus is in Supabase if Aglaea ever needs deeper read.
            txt = (t.get("text") or "")[:_TRANSCRIPT_CHAR_CAP]
            parts.append(txt)
    else:
        parts.append("=== TRANSCRIPTS ===\n(no client transcripts available — voice signal limited to LinkedIn posts)")

    # Grounded per-post bundle — appended after the aggregate blocks
    # above. Where the aggregates give pattern-level signal, the
    # bundle gives literal draft→comment pairings so you can cite the
    # exact rejection (e.g., "Andrew rejected this framing last
    # Tuesday") rather than extrapolating from aggregate feedback.
    # Rejected drafts are the highest-weight signal in the bundle.
    pb = (signal.get("post_bundle") or "").strip()
    if pb:
        parts.append("")
        parts.append(pb)

    return "\n".join(parts)


def _call_model(draft_text: str, signal: dict[str, Any]) -> dict[str, Any]:
    """One Claude call. Structured JSON out.

    Routes to Claude CLI (Max subscription) when ``AMPHOREUS_USE_CLI=true``,
    falls through to the Anthropic API otherwise. The CLI path uses the
    same system prompt + user text via ``cli_single_shot`` so the contract
    (JSON string out) is preserved for the parsing below.
    """
    signal_block = _render_signal_block(signal)
    user_text = (
        "Evaluate whether the TARGET FOC USER below would publish the DRAFT as-is.\n\n"
        f"{signal_block}\n\n"
        "=== DRAFT ===\n"
        f"{draft_text}\n"
        "=== END DRAFT ===\n\n"
        "Respond with ONLY the JSON object. No prose."
    )

    raw = ""
    resp = None  # only bound on the API path; checked before cost calc below
    try:
        from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli, cli_single_shot as _cli_ss
    except ImportError:
        _use_cli = lambda: False  # type: ignore
        _cli_ss = None             # type: ignore

    if _use_cli() and _cli_ss is not None:
        # CLI uses model aliases ("haiku"/"sonnet"/"opus") not full ids.
        # Map the full id by picking the shortest unique substring.
        cli_model = "sonnet" if "sonnet" in _AGLAEA_MODEL else (
            "opus" if "opus" in _AGLAEA_MODEL else (
                "haiku" if "haiku" in _AGLAEA_MODEL else "sonnet"
            )
        )
        raw = (_cli_ss(
            user_text,
            model=cli_model,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=_AGLAEA_MAX_TOKENS,
            timeout=120,
        ) or "").strip()
    else:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_AGLAEA_MODEL,
            max_tokens=_AGLAEA_MAX_TOKENS,
            system=[
                {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": user_text}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                raw += block.text
        raw = raw.strip()

    # Strip fenced code block if model wrapped it.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[Aglaea] non-JSON response; returning raw as summary")
        parsed = {"score": None, "summary": raw[:200], "flagged_spans": []}

    # Normalise + stamp bookkeeping
    score = parsed.get("score")
    try:
        score = int(score) if score is not None else None
    except Exception:
        score = None
    spans = parsed.get("flagged_spans") or []
    if not isinstance(spans, list):
        spans = []

    # Rough cost estimate (Sonnet 4.5 pricing). Only available on the
    # API path — the CLI path rides the Max subscription and has no
    # per-request cost we can measure here, so leave cost=0.0.
    cost = 0.0
    usage = getattr(resp, "usage", None) if resp is not None else None
    if usage is not None:
        # $3/MT input + $15/MT output (subject to model)
        in_t = getattr(usage, "input_tokens", 0) or 0
        out_t = getattr(usage, "output_tokens", 0) or 0
        cost = round(in_t * 3e-6 + out_t * 15e-6, 5)

    return {
        "score":         score,
        "summary":       (parsed.get("summary") or "")[:200],
        "flagged_spans": spans,
        "_model":        _AGLAEA_MODEL,
        "_cost_usd":     cost,
    }
