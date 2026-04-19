"""Direct-access reads of Jacquard's data: Supabase queries + GCS blob downloads.

Bypasses Jacquard's ``virio-api/workspace`` HTTP layer by talking to the
same Supabase project + GCS bucket directly. Used by Stelle when we want
full independence from Jacquard's runtime (no dependency on their API
being up, no shared run-token lifecycle).

What this module provides, matched to Jacquard's workspace endpoints:

- ``list_foc_users(company_id)``          → slugs of FOC users in a company
- ``fetch_meeting_transcripts(user_id, email, is_internal)``
                                          → per-meeting ``{filename, content}``.
                                            Content fetched from GCS when
                                            ``gcs_transcript_url`` is set,
                                            else from inline ``transcript_text``.
- ``fetch_parallel_research(company_id, user_id)``
                                          → company + person research files.
- ``fetch_icp_data(user)``                → posts/reactions/comments/
                                            profiles/work_experiences/client_info
                                            as JSON blobs.
- ``fetch_latest_icp_report(company_id, user_id)``
                                          → the most recent ICP report JSON.
- ``fetch_context_files(company_id)``     → account.md (generated) +
                                            uploaded brand docs. File bodies
                                            come from ``extracted_text`` or
                                            GCS at ``gcs_url``.
- ``fetch_tone_references(user_id)``      → style/voice reference files.
- ``fetch_edit_history(user_id, limit)``  → published drafts with earliest
                                            snapshot + final text + comments
                                            (the feedback signal).
- ``fetch_published_posts(user_id)``      → drafts with status='published'.
- ``fetch_trigger_log(company_id)``       → trigger_log + company_events
                                            merged chronologically.
- ``fetch_tasks(company_id)``             → pending review / task rows.

Every function returns plain Python dicts/lists matching the shapes
Jacquard's workspace-builder.ts produces. Drop-in compatible when we
want to swap an HTTP call for a direct call.

Credentials:
- ``GCS_CREDENTIALS_B64``: base64-encoded service account JSON.
- ``GCS_BUCKET``: default bucket name (fallback: ``lino-meeting-transcripts``).
- ``SUPABASE_URL`` / ``SUPABASE_KEY``: the Amphoreus-side Supabase config,
  which happens to point at the same project Jacquard uses.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

MAX_MEETING_TRANSCRIPTS = 5   # same default as Jacquard


def is_direct_configured() -> bool:
    """True when we have everything for GCS + Supabase direct reads.

    When True, callers can skip Jacquard's virio-api entirely. When False,
    fall back to the HTTP workspace client (``lineage_fs_client``).
    """
    return bool(
        os.environ.get("GCS_CREDENTIALS_B64", "").strip()
        and os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_KEY", "").strip()
    )


# ---------------------------------------------------------------------------
# GCS client (lazy, memoized)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _gcs_storage_client():
    """Singleton google-cloud-storage client authed from ``GCS_CREDENTIALS_B64``.

    Decodes the base64 JSON once, feeds it to ``service_account.Credentials``.
    No filesystem write — keeps the key out of /tmp.
    """
    from google.cloud import storage
    from google.oauth2 import service_account

    b64 = os.environ.get("GCS_CREDENTIALS_B64", "").strip()
    if not b64:
        raise RuntimeError("GCS_CREDENTIALS_B64 is not set")

    # Strip optional surrounding quotes (some env parsers keep them).
    b64 = b64.strip('"').strip("'")
    try:
        info = json.loads(base64.b64decode(b64).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"GCS_CREDENTIALS_B64 could not be decoded: {exc}") from exc

    creds = service_account.Credentials.from_service_account_info(info)
    return storage.Client(project=info.get("project_id"), credentials=creds)


_GS_URL_RE = re.compile(r"^gs://([^/]+)/(.+)$")


def _parse_gcs_url(url: str) -> tuple[str, str] | None:
    """Parse ``gs://bucket/path`` → ``(bucket, path)``; None if malformed."""
    if not url:
        return None
    match = _GS_URL_RE.match(url.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def download_gcs_text(gcs_url: str, timeout: float = 30.0) -> str | None:
    """Download a GCS object as UTF-8 text, or None on any failure.

    Matches Jacquard's ``gcs.downloadText`` semantics — silent failure
    (returns None) rather than raising, so callers can treat missing
    blobs as "no content" without a try/except at every call site.
    """
    parsed = _parse_gcs_url(gcs_url)
    if not parsed:
        logger.debug("[jacquard_direct] malformed gcs_url: %r", gcs_url)
        return None
    bucket_name, blob_path = parsed
    try:
        client = _gcs_storage_client()
        blob = client.bucket(bucket_name).blob(blob_path)
        buf = io.BytesIO()
        blob.download_to_file(buf, timeout=timeout)
        return buf.getvalue().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("[jacquard_direct] GCS download failed (%s): %s", gcs_url, exc)
        return None


# ---------------------------------------------------------------------------
# Supabase client (reuses Amphoreus's existing singleton)
# ---------------------------------------------------------------------------

def _sb():
    """Return Amphoreus's Supabase client. Same project as Jacquard's."""
    from backend.src.db.supabase_client import get_supabase
    return get_supabase()


# ---------------------------------------------------------------------------
# FOC users (company_id → list of users with slug)
# ---------------------------------------------------------------------------

def _slugify_user(first: str | None, last: str | None) -> str:
    name = " ".join(filter(None, [first, last])).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return slug


def list_foc_users(company_id: str) -> list[dict[str, Any]]:
    """Return the FOC users for a company.

    Each dict: ``{id, email, first_name, last_name, is_internal, linkedin_url,
    company_id, company_name, posts_per_month, slug}``.
    """
    if not company_id:
        return []
    sb = _sb()
    users = (
        sb.table("users")
        .select(
            "id, email, first_name, last_name, is_internal, linkedin_url, "
            "company_id, posts_per_month"
        )
        .eq("company_id", company_id)
        .execute()
        .data
        or []
    )
    # Pull the company name once for all users in this company.
    company_row = (
        sb.table("user_companies")
        .select("id, name, assigned_am")
        .eq("id", company_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    company_name = company_row[0].get("name") if company_row else ""
    out: list[dict[str, Any]] = []
    for u in users:
        out.append({
            **u,
            "slug": _slugify_user(u.get("first_name"), u.get("last_name")),
            "company_name": company_name,
        })
    return out


def resolve_user_by_slug(company_id: str, slug: str) -> dict[str, Any] | None:
    """Find the FOC user in ``company_id`` whose slug matches."""
    for u in list_foc_users(company_id):
        if u.get("slug") == slug:
            return u
    return None


# ---------------------------------------------------------------------------
# Meetings / transcripts  (Supabase + GCS)
# ---------------------------------------------------------------------------

_MEETING_COLUMNS = (
    "provider_transcript_id, name, start_time, duration_seconds, "
    "gcs_transcript_url, transcript_text, recorded_by"
)


def _exclude_client_meetings(transcript_ids: list[str]) -> set[str]:
    """For internal users — find meetings where any participant is non-internal
    (client-facing), and exclude those. Mirrors Jacquard's excludeClientMeetings.
    """
    if not transcript_ids:
        return set()
    sb = _sb()
    rows = (
        sb.table("meeting_participants")
        .select("provider_transcript_id, is_internal")
        .in_("provider_transcript_id", transcript_ids)
        .execute()
        .data
        or []
    )
    client_ids: set[str] = set()
    for r in rows:
        if r.get("is_internal") is False and r.get("provider_transcript_id"):
            client_ids.add(r["provider_transcript_id"])
    return client_ids


def _meeting_filename(row: dict[str, Any]) -> str:
    """``2026-04-15-call-with-X.md`` — matches Jacquard's filename convention."""
    start = row.get("start_time") or ""
    date_part = start[:10] if start else "unknown-date"
    name = (row.get("name") or "untitled").strip()
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "untitled"
    return f"{date_part}-{slug}.md"


def _meeting_content(row: dict[str, Any]) -> str:
    """Return the transcript body: inline ``transcript_text`` preferred,
    falls back to GCS blob at ``gcs_transcript_url``. Header with metadata."""
    header_lines = [
        f"# {row.get('name') or 'Untitled meeting'}",
        "",
        f"_start_time: {row.get('start_time')}_",
    ]
    if row.get("duration_seconds"):
        header_lines.append(f"_duration_seconds: {row['duration_seconds']}_")
    header_lines.append("")

    inline = row.get("transcript_text")
    if inline:
        return "\n".join(header_lines) + inline
    gcs = row.get("gcs_transcript_url")
    if gcs:
        body = download_gcs_text(gcs)
        if body is not None:
            return "\n".join(header_lines) + body
    return "\n".join(header_lines) + "_(transcript unavailable)_"


def fetch_meeting_transcripts(
    user_id: str,
    email: str | None,
    is_internal: bool = False,
) -> list[dict[str, str]]:
    """Return up to ``MAX_MEETING_TRANSCRIPTS`` meeting transcripts the user
    participated in (or recorded, as fallback). Shape mirrors
    ``fetchMeetingTranscripts`` in Jacquard's data-gathering.ts.
    """
    if not user_id:
        return []
    sb = _sb()

    transcript_ids: list[str] = []
    transcript_id_set: set[str] = set()
    if email:
        rows = (
            sb.table("meeting_participants")
            .select("provider_transcript_id")
            .eq("email", email)
            .execute()
            .data
            or []
        )
        for r in rows:
            pid = r.get("provider_transcript_id")
            if pid and pid not in transcript_id_set:
                transcript_id_set.add(pid)
                transcript_ids.append(pid)

    client_meeting_ids: set[str] | None = None
    if is_internal and transcript_ids:
        client_meeting_ids = _exclude_client_meetings(transcript_ids)
        transcript_ids = [t for t in transcript_ids if t not in client_meeting_ids]

    meetings: list[dict[str, Any]] = []
    if transcript_ids:
        meetings = (
            sb.table("meetings")
            .select(_MEETING_COLUMNS)
            .in_("provider_transcript_id", transcript_ids)
            .order("start_time", desc=True)
            .limit(MAX_MEETING_TRANSCRIPTS)
            .execute()
            .data
            or []
        )

    if not meetings:
        # Fallback: meetings the user recorded themselves.
        meetings = (
            sb.table("meetings")
            .select(_MEETING_COLUMNS)
            .eq("recorded_by", user_id)
            .order("start_time", desc=True)
            .limit(MAX_MEETING_TRANSCRIPTS)
            .execute()
            .data
            or []
        )
        if is_internal and meetings:
            excl = _exclude_client_meetings(
                [m.get("provider_transcript_id") for m in meetings if m.get("provider_transcript_id")]
            )
            meetings = [m for m in meetings if m.get("provider_transcript_id") not in excl]

    out: list[dict[str, str]] = []
    for m in meetings:
        # Only keep meetings that have either inline text or a GCS URL —
        # otherwise the file body would just be a stub.
        if not m.get("transcript_text") and not m.get("gcs_transcript_url"):
            continue
        out.append({"filename": _meeting_filename(m), "content": _meeting_content(m)})
    return out


# ---------------------------------------------------------------------------
# Parallel research
# ---------------------------------------------------------------------------

def fetch_parallel_research(company_id: str, user_id: str) -> list[dict[str, str]]:
    """Return company-research + person-research files if present."""
    if not company_id and not user_id:
        return []
    sb = _sb()
    files: list[dict[str, str]] = []

    if company_id:
        rows = (
            sb.table("parallel_research_results")
            .select("output, created_at")
            .eq("company_id", company_id)
            .eq("research_type", "company")
            .is_("error", "null")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        if rows:
            output = rows[0].get("output")
            content = output if isinstance(output, str) else json.dumps(output or {}, indent=2)
            files.append({"filename": "company-research.md", "content": content})

    if user_id:
        rows = (
            sb.table("parallel_research_results")
            .select("output, created_at")
            .eq("user_id", user_id)
            .eq("research_type", "person")
            .is_("error", "null")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        if rows:
            output = rows[0].get("output")
            content = output if isinstance(output, str) else json.dumps(output or {}, indent=2)
            files.append({"filename": "person-research.md", "content": content})

    return files


# ---------------------------------------------------------------------------
# Context files (company-scoped brand docs + auto-generated account.md)
# ---------------------------------------------------------------------------

def fetch_context_files(
    company_id: str,
    user: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return the context/ directory contents for a user.

    - ``account.md`` — auto-generated {name, company, posts_per_month, slack channels}.
    - Every row in ``context_files`` for this company — content from
      ``extracted_text`` if present, else downloaded from GCS.
    """
    if not company_id:
        return []
    sb = _sb()

    out: list[dict[str, str]] = []

    # account.md
    if user:
        name = " ".join(filter(None, [user.get("first_name"), user.get("last_name")]))
        slack_rows = (
            sb.table("slack_channels")
            .select("provider_id, name, type")
            .eq("foc_user_id", user.get("id"))
            .execute()
            .data
            or []
        )
        slack_lines = [
            f"Slack channel: {ch.get('provider_id')} ({ch.get('name')})"
            for ch in slack_rows
        ]
        account_lines = [f"# {name}", f"Company: {user.get('company_name', '')}"]
        if user.get("posts_per_month"):
            account_lines.append(f"Posts per month target: {user['posts_per_month']}")
        account_lines.extend(slack_lines)
        out.append({"filename": "account.md", "content": "\n".join(account_lines)})

    # Uploaded context files
    files = (
        sb.table("context_files")
        .select("filename, extracted_text, gcs_url")
        .eq("company_id", company_id)
        .execute()
        .data
        or []
    )
    for f in files:
        filename = f.get("filename") or "untitled.txt"
        content = f.get("extracted_text")
        if not content and f.get("gcs_url"):
            content = download_gcs_text(f["gcs_url"]) or ""
        out.append({"filename": filename, "content": content or ""})

    return out


# ---------------------------------------------------------------------------
# ICP engagement data
# ---------------------------------------------------------------------------

def _linkedin_username_from_url(url: str | None) -> str | None:
    """Extract 'lukasheuer' from 'https://www.linkedin.com/in/lukasheuer/' etc."""
    if not url:
        return None
    match = re.search(r"/in/([^/?#]+)", url)
    if not match:
        return None
    return match.group(1).strip().lower()


def fetch_icp_data(user: dict[str, Any]) -> dict[str, str]:
    """Return the engagement/ directory as ``{filename: json_content_str}``.

    Files: ``posts.json``, ``reactions.json``, ``comments.json``,
    ``profiles.json``, ``client_info.json``.

    Every blob is JSON-serialized. Callers expect strings and parse themselves.
    Missing data yields empty arrays rather than erroring.

    Column schema matches Jacquard's actual linkedin_posts / linkedin_reactions /
    linkedin_comments / linkedin_profiles tables (real column names, not the
    ones I guessed from Lineage's workspace-builder reads).
    """
    files: dict[str, str] = {}
    if not user or not user.get("linkedin_url"):
        return files

    sb = _sb()
    user_id = user.get("id")
    linkedin_url = user.get("linkedin_url")
    username = _linkedin_username_from_url(linkedin_url)

    # Posts authored by this user — joined on creator_username.
    posts = (
        sb.table("linkedin_posts")
        .select(
            "id, provider_urn, post_url, posted_at, post_text, hook, "
            "total_reactions, total_comments, total_reposts, "
            "like_reactions, love_reactions, support_reactions, "
            "insight_reactions, celebrate_reactions, funny_reactions, "
            "engagement_score, is_outlier, quality_score, "
            "topic_tags, hook_type, format_archetype, tone, "
            "is_tofu, is_mofu, is_bofu, abm_category, target_persona"
        )
        .eq("creator_username", username or "")
        .order("posted_at", desc=True)
        .limit(200)
        .execute()
        .data
        or []
    )
    files["posts.json"] = json.dumps(posts, default=str, indent=2)

    # Reactions + comments are keyed by provider_post_urn, not numeric id.
    post_urns = [p.get("provider_urn") for p in posts if p.get("provider_urn")]

    reactions: list[dict[str, Any]] = []
    if post_urns:
        reactions = (
            sb.table("linkedin_reactions")
            .select(
                "id, provider_post_urn, provider_profile_urn, type, "
                "reacted_at, reactor_name, reactor_headline"
            )
            .in_("provider_post_urn", post_urns)
            .limit(5000)
            .execute()
            .data
            or []
        )
    files["reactions.json"] = json.dumps(reactions, default=str, indent=2)

    comments: list[dict[str, Any]] = []
    if post_urns:
        comments = (
            sb.table("linkedin_comments")
            .select(
                "id, provider_post_urn, provider_profile_urn, text, "
                "commented_at, commenter_name, commenter_headline"
            )
            .in_("provider_post_urn", post_urns)
            .limit(2000)
            .execute()
            .data
            or []
        )
    files["comments.json"] = json.dumps(comments, default=str, indent=2)

    # Reactor + commenter profiles — join on provider_urn.
    engager_urns = sorted({
        *(r.get("provider_profile_urn") for r in reactions if r.get("provider_profile_urn")),
        *(c.get("provider_profile_urn") for c in comments if c.get("provider_profile_urn")),
    })

    profiles: list[dict[str, Any]] = []
    if engager_urns:
        profiles = (
            sb.table("linkedin_profiles")
            .select(
                "provider_urn, username, first_name, last_name, headline, "
                "about, location_full, location_country, location_city, "
                "follower_count, connection_count, positions_text, "
                "engagement_scores, skills, has_education, has_work_experience"
            )
            .in_("provider_urn", engager_urns)
            .execute()
            .data
            or []
        )
    files["profiles.json"] = json.dumps(profiles, default=str, indent=2)

    files["client_info.json"] = json.dumps({
        "user_id": user_id,
        "name": " ".join(filter(None, [user.get("first_name"), user.get("last_name")])),
        "email": user.get("email"),
        "linkedin_url": linkedin_url,
        "linkedin_username": username,
        "company_name": user.get("company_name"),
        "is_internal": user.get("is_internal"),
        "n_posts_tracked": len(posts),
        "n_reactions_tracked": len(reactions),
        "n_comments_tracked": len(comments),
        "n_unique_engagers": len(engager_urns),
    }, default=str, indent=2)

    return files


# ---------------------------------------------------------------------------
# ICP report (latest)
# ---------------------------------------------------------------------------

def fetch_latest_icp_report(company_id: str, user_id: str) -> dict[str, str] | None:
    """Return the most recent ICP report as ``{filename, content}`` or None.

    Real schema: ``reports`` table has {id, company_id, user_id, run_id,
    report_type, result_json, pdf_gcs_url, csv_gcs_url, title, summary,
    status, created_at, completed_at}. We filter to report_type='icp' and
    status='completed', grab the most recent, serialize result_json.
    """
    if not company_id or not user_id:
        return None
    sb = _sb()
    try:
        rows = (
            sb.table("reports")
            .select("id, report_type, result_json, title, summary, status, completed_at")
            .eq("company_id", company_id)
            .eq("user_id", user_id)
            .eq("report_type", "icp")
            .eq("status", "completed")
            .order("completed_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.debug("[jacquard_direct] fetch_latest_icp_report: %s", exc)
        return None
    if not rows:
        return None
    row = rows[0]
    body = row.get("result_json")
    if isinstance(body, (dict, list)):
        body = json.dumps(body, default=str, indent=2)
    return {
        "filename": f"icp-report-{(row.get('id') or '')[:8]}.json",
        "content": body or "",
    }


# ---------------------------------------------------------------------------
# Tone references
# ---------------------------------------------------------------------------

def fetch_tone_references(user_id: str) -> list[dict[str, str]]:
    """Per-user style/voice exemplars.

    Real schema: tone_references has {id, user_id, source_type, url, raw_text,
    note, preferred, calibration_pair}. No filename column — we synthesize one.
    """
    if not user_id:
        return []
    sb = _sb()
    rows = (
        sb.table("tone_references")
        .select("id, source_type, url, raw_text, note, preferred, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
        .data
        or []
    )
    out: list[dict[str, str]] = []
    for i, r in enumerate(rows, 1):
        source = r.get("source_type") or "tone"
        marker = "preferred" if r.get("preferred") else source
        filename = f"{i:02d}-{marker}-{(r.get('id') or '')[:8]}.md"
        lines = [f"# Tone reference {i}: {marker}"]
        if r.get("url"):
            lines.append(f"_source_url: {r['url']}_")
        if r.get("note"):
            lines.append(f"\n**Note:** {r['note']}")
        lines.append("")
        lines.append(r.get("raw_text") or "")
        out.append({"filename": filename, "content": "\n".join(lines)})
    return out


# ---------------------------------------------------------------------------
# Draft edit history  (feedback signal)
# ---------------------------------------------------------------------------

def fetch_edit_history(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return published drafts with their earliest snapshot + final content +
    operator comments, rendered into markdown. Mirrors the ``edits/`` mount
    in Jacquard's workspace-builder (the feedback signal Stelle reads to
    learn how operators revise her output).
    """
    if not user_id:
        return []
    sb = _sb()
    drafts = (
        sb.table("drafts")
        .select("id, title, content, created_at, updated_at")
        .eq("user_id", user_id)
        .eq("status", "published")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )

    out: list[dict[str, Any]] = []
    for d in drafts:
        draft_id = d.get("id")
        snaps = (
            sb.table("draft_snapshots")
            .select("content_preview, created_at, edit_count, word_count_delta")
            .eq("draft_id", draft_id)
            .order("created_at", desc=False)
            .limit(50)
            .execute()
            .data
            or []
        )
        comments = (
            sb.table("draft_comments")
            .select(
                "id, user_id, selection_start, selection_end, selected_text, "
                "content, resolved, parent_comment_id, created_at"
            )
            .eq("draft_id", draft_id)
            .order("created_at", desc=False)
            .limit(200)
            .execute()
            .data
            or []
        )
        first_snap = snaps[0].get("content_preview") if snaps else ""
        date_str = (d.get("updated_at") or d.get("created_at") or "")[:10]
        title = d.get("title") or "untitled"
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "untitled"
        filename = f"{date_str}-{slug}-{draft_id}.md"

        lines: list[str] = [
            f"# {title}",
            "",
            f"_draft_id: {draft_id}_",
            f"_published: {d.get('updated_at')}_",
            "",
            "## Original (earliest snapshot)",
            "",
            first_snap or "_(no snapshot recorded)_",
            "",
            "## Final (published)",
            "",
            d.get("content") or "",
            "",
            "## Review comments",
            "",
        ]
        if not comments:
            lines.append("_No operator comments on this draft._")
        else:
            for c in comments:
                author_id = c.get("user_id") or "unknown"
                when = c.get("created_at")
                sel = (c.get("selected_text") or "")[:80]
                resolved = " _(resolved)_" if c.get("resolved") else ""
                header = f"- {author_id} @ {when}{resolved}"
                if sel:
                    header += f' — inline on "{sel}"'
                lines.append(header)
                body = (c.get("content") or "").split("\n")
                lines.extend(f"    {bl}" for bl in body)
                lines.append("")

        out.append({"filename": filename, "content": "\n".join(lines)})
    return out


# ---------------------------------------------------------------------------
# Published posts
# ---------------------------------------------------------------------------

def build_post_history_digest(user: dict[str, Any], top_n: int = 10) -> str:
    """Render the client's top-N posts by reactions as a single markdown
    file — Stelle's primary "what works for this audience" signal.

    Replaces the old memory/<company>/post-history.md that used to come
    from ruan_mei's scored observations. Same semantic: the best-performing
    past posts, with full text + raw engagement numbers + per-reaction
    breakdown, so Stelle can learn patterns from evidence without us
    pre-digesting into categorical tags.
    """
    if not user or not user.get("linkedin_url"):
        return (
            "# Post history\n\n"
            "(no LinkedIn URL configured for this user — engagement data unavailable)\n"
        )
    sb = _sb()
    username = _linkedin_username_from_url(user.get("linkedin_url"))
    if not username:
        return "# Post history\n\n(could not parse username from linkedin_url)\n"

    try:
        posts = (
            sb.table("linkedin_posts")
            .select(
                "posted_at, post_text, hook, post_url, "
                "total_reactions, total_comments, total_reposts, "
                "like_reactions, love_reactions, support_reactions, "
                "insight_reactions, celebrate_reactions, funny_reactions, "
                "engagement_score, is_outlier, topic_tags, hook_type, format_archetype"
            )
            .eq("creator_username", username)
            .order("total_reactions", desc=True)
            .limit(top_n)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.warning("[jacquard_direct] build_post_history_digest failed: %s", exc)
        return f"# Post history\n\n(fetch failed: {exc})\n"

    if not posts:
        return "# Post history\n\n(no scored posts for this user yet)\n"

    name = " ".join(filter(None, [user.get("first_name"), user.get("last_name")])) or username

    lines: list[str] = []
    lines.append(f"# Post history — {name} — top {len(posts)} performers")
    lines.append("")
    lines.append(
        f"These are the {len(posts)} highest-reacted posts from this user's "
        f"LinkedIn history, sorted by total_reactions. Use them to learn what "
        f"structures, topics, and tones land with this audience. Full text is "
        f"included — NOT a summary — so you can see what actually worked."
    )
    lines.append("")
    lines.append("Per-reaction breakdown: `like / love / support / insight / celebrate / funny`.")
    lines.append("")

    for i, p in enumerate(posts, 1):
        posted = (p.get("posted_at") or "")[:10]
        reacts = p.get("total_reactions") or 0
        comms = p.get("total_comments") or 0
        reps = p.get("total_reposts") or 0
        breakdown = (
            f"{p.get('like_reactions') or 0}/"
            f"{p.get('love_reactions') or 0}/"
            f"{p.get('support_reactions') or 0}/"
            f"{p.get('insight_reactions') or 0}/"
            f"{p.get('celebrate_reactions') or 0}/"
            f"{p.get('funny_reactions') or 0}"
        )
        outlier_mark = " 🔥 OUTLIER" if p.get("is_outlier") else ""
        eng_score = p.get("engagement_score")

        lines.append(f"## {i}. {reacts} reactions · {comms} comments · {reps} reposts · {posted}{outlier_mark}")
        lines.append("")
        lines.append(f"- reaction breakdown: `{breakdown}`")
        if eng_score is not None:
            lines.append(f"- engagement_score (Jacquard): `{eng_score}`")
        tags = p.get("topic_tags") or []
        if tags:
            lines.append(f"- topic_tags: `{tags}`")
        ht = p.get("hook_type")
        fa = p.get("format_archetype")
        if ht or fa:
            lines.append(f"- hook_type: `{ht}` · format: `{fa}`")
        if p.get("post_url"):
            lines.append(f"- link: {p['post_url']}")
        lines.append("")
        lines.append("```")
        lines.append(p.get("post_text") or "")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def fetch_profile_md(user: dict[str, Any]) -> str | None:
    """Render the user's LinkedIn profile as markdown.

    Replaces the old memory/<company>/profile.md (which came from APImaestro).
    Source: linkedin_profiles row matched by username.
    """
    username = _linkedin_username_from_url(user.get("linkedin_url"))
    if not username:
        return None
    sb = _sb()
    try:
        rows = (
            sb.table("linkedin_profiles")
            .select(
                "username, first_name, last_name, headline, about, "
                "location_full, location_country, location_city, metro_area, "
                "follower_count, connection_count, positions_text, "
                "skills, languages, certifications, projects, "
                "total_posts, last_updated_at"
            )
            .eq("username", username)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.debug("[jacquard_direct] fetch_profile_md: %s", exc)
        return None
    if not rows:
        return None
    p = rows[0]
    name = " ".join(filter(None, [p.get("first_name"), p.get("last_name")])) or username

    lines: list[str] = [
        f"# {name}",
        "",
        f"_username: {p.get('username')}_",
        f"_last_updated: {p.get('last_updated_at')}_",
        "",
    ]
    if p.get("headline"):
        lines.extend(["## Headline", "", p["headline"], ""])
    if p.get("location_full"):
        lines.extend(["## Location", "", p["location_full"], ""])
    counts = []
    if p.get("follower_count") is not None:
        counts.append(f"followers: {p['follower_count']}")
    if p.get("connection_count") is not None:
        counts.append(f"connections: {p['connection_count']}")
    if p.get("total_posts") is not None:
        counts.append(f"total posts: {p['total_posts']}")
    if counts:
        lines.extend(["## Counts", "", " · ".join(counts), ""])
    if p.get("about"):
        lines.extend(["## About", "", p["about"], ""])
    if p.get("positions_text"):
        lines.extend(["## Positions", "", p["positions_text"], ""])
    if p.get("skills"):
        lines.extend(["## Skills", "", json.dumps(p["skills"], indent=2), ""])
    return "\n".join(lines)


def fetch_published_posts(user_id: str, limit: int = 100) -> list[dict[str, str]]:
    """Posts the user has published on LinkedIn. Status-filtered drafts table."""
    if not user_id:
        return []
    sb = _sb()
    rows = (
        sb.table("drafts")
        .select("id, title, content, scheduled_date, updated_at")
        .eq("user_id", user_id)
        .eq("status", "published")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )
    out: list[dict[str, str]] = []
    for d in rows:
        draft_id = d.get("id")
        title = d.get("title") or "untitled"
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "untitled"
        filename = f"{(d.get('scheduled_date') or d.get('updated_at') or '')[:10]}-{slug}-{draft_id}.md"
        out.append({
            "filename": filename,
            "content": f"# {title}\n\n{d.get('content') or ''}",
        })
    return out


# ---------------------------------------------------------------------------
# Trigger log + company events  (conversation history replay)
# ---------------------------------------------------------------------------

def fetch_trigger_log(company_id: str, limit: int = 500) -> str:
    """Return ``trigger_log`` + ``company_events`` for this company merged
    chronologically, as newline-delimited JSON (matches Jacquard's
    ``conversations/trigger-log.jsonl`` mount).
    """
    if not company_id:
        return ""
    sb = _sb()
    trigger = (
        sb.table("trigger_log")
        .select("*")
        .eq("company_id", company_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
        .data
        or []
    )
    events = (
        sb.table("company_events")
        .select("*")
        .eq("company_id", company_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
        .data
        or []
    )
    merged = sorted(
        trigger + events,
        key=lambda r: r.get("created_at") or "",
    )
    return "\n".join(json.dumps(r, default=str) for r in merged)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def fetch_tasks(company_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Open review tasks for a company — what the CE queue is currently chewing."""
    if not company_id:
        return []
    sb = _sb()
    rows = (
        sb.table("tasks")
        .select("id, type, status, payload, created_at, resolved_at")
        .eq("company_id", company_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )
    return rows


# ---------------------------------------------------------------------------
# Workspace population — used by CLI mode, where Claude CLI's native
# filesystem tools read local disk instead of going through
# ``lineage_fs_client``. We pre-fetch everything into a local directory
# laid out the way the LINEAGE_DIRECTIVES prompt describes so Stelle sees
# the same paths whether she's running direct-API or CLI.
# ---------------------------------------------------------------------------


def _write_files(base: "Path", items: list[dict[str, str]]) -> int:
    """Write each {filename, content} into base. Returns count written."""
    from pathlib import Path as _P
    base = _P(base)
    if not items:
        return 0
    base.mkdir(parents=True, exist_ok=True)
    n = 0
    for item in items:
        name = (item.get("filename") or "").strip()
        content = item.get("content") or ""
        if not name:
            continue
        # Filesystem-safe: strip any path separators that slipped into
        # Jacquard's filename column (shouldn't happen but be defensive).
        safe = name.replace("/", "_").replace("\\", "_")
        (base / safe).write_text(content, encoding="utf-8")
        n += 1
    return n


def populate_lineage_workspace(
    root: "Path",
    company_id: str,
    target_user_slug: str | None = None,
) -> dict[str, int]:
    """Pre-populate ``root`` with Jacquard's workspace data so a CLI-mode
    Stelle can read it via Claude CLI's native filesystem tools.

    Layout matches Jacquard's workspace-builder.ts mounts so Stelle's
    Lineage-mode prompt directives work unchanged:

        {user-slug}/transcripts/*.md
        {user-slug}/research/*.md
        {user-slug}/engagement/*.json
        {user-slug}/context/*.md
        {user-slug}/reports/*.json
        {user-slug}/edits/*.md
        {user-slug}/tone/*.md
        {user-slug}/posts/published/*.md
        conversations/trigger-log.jsonl

    Args:
        root: workspace directory. Created if missing.
        company_id: Jacquard ``user_companies.id`` UUID.
        target_user_slug: if provided, only that user's subtree is
            populated (USER-TARGETED mode). Otherwise all FOC users of
            the company are populated (COMPANY-WIDE mode).

    Returns: dict of per-mount file counts for logging/debugging.
    """
    from pathlib import Path as _P
    root = _P(root)
    root.mkdir(parents=True, exist_ok=True)

    users = list_foc_users(company_id)
    if target_user_slug:
        users = [u for u in users if u.get("slug") == target_user_slug]
    if not users:
        raise RuntimeError(
            f"No FOC users found for company_id={company_id} "
            f"(target_slug={target_user_slug!r})"
        )

    counts: dict[str, int] = {}

    def _safe(key: str, fn, *a, **kw):
        """Run a fetcher; on any failure, log + return empty. Keeps a
        schema drift or permissions issue in ONE mount from poisoning
        the whole workspace."""
        try:
            return fn(*a, **kw)
        except Exception as exc:
            logger.warning("[populate_lineage_workspace] %s failed: %s", key, exc)
            counts[key] = 0
            return None

    for user in users:
        slug = user.get("slug") or "unknown"
        user_root = root / slug

        # Transcripts — the big one, GCS-backed.
        items = _safe(
            f"{slug}/transcripts",
            fetch_meeting_transcripts,
            user.get("id", ""),
            user.get("email"),
            user.get("is_internal", False),
        )
        counts[f"{slug}/transcripts"] = _write_files(user_root / "transcripts", items or [])

        # Research
        items = _safe(
            f"{slug}/research",
            fetch_parallel_research, company_id, user.get("id", ""),
        )
        counts[f"{slug}/research"] = _write_files(user_root / "research", items or [])

        # Context (account.md + uploaded brand docs)
        items = _safe(
            f"{slug}/context",
            fetch_context_files, company_id, user,
        )
        counts[f"{slug}/context"] = _write_files(user_root / "context", items or [])

        # Engagement — structured JSON blobs.
        eng_files = _safe(f"{slug}/engagement", fetch_icp_data, user) or {}
        if eng_files:
            eng_dir = user_root / "engagement"
            eng_dir.mkdir(parents=True, exist_ok=True)
            for fname, content in eng_files.items():
                (eng_dir / fname).write_text(content, encoding="utf-8")
            counts[f"{slug}/engagement"] = len(eng_files)
        else:
            counts.setdefault(f"{slug}/engagement", 0)

        # Latest ICP report
        report = _safe(
            f"{slug}/reports", fetch_latest_icp_report, company_id, user.get("id", ""),
        )
        if report:
            rpt_dir = user_root / "reports"
            rpt_dir.mkdir(parents=True, exist_ok=True)
            (rpt_dir / report["filename"]).write_text(report["content"], encoding="utf-8")
            counts[f"{slug}/reports"] = 1
        else:
            counts.setdefault(f"{slug}/reports", 0)

        # Tone references
        items = _safe(
            f"{slug}/tone", fetch_tone_references, user.get("id", ""),
        )
        counts[f"{slug}/tone"] = _write_files(user_root / "tone", items or [])

        # Edit history (feedback signal)
        items = _safe(
            f"{slug}/edits", fetch_edit_history, user.get("id", ""), limit=50,
        )
        counts[f"{slug}/edits"] = _write_files(user_root / "edits", items or [])

        # Published posts
        items = _safe(
            f"{slug}/posts/published", fetch_published_posts, user.get("id", ""), limit=100,
        )
        counts[f"{slug}/posts/published"] = _write_files(
            user_root / "posts" / "published", items or [],
        )

        # post-history.md — top-N performers with full text + engagement.
        # This is Stelle's critical "what works for this audience" signal.
        digest = _safe(f"{slug}/post-history.md", build_post_history_digest, user, top_n=10)
        if digest:
            (user_root / "post-history.md").write_text(digest, encoding="utf-8")
            counts[f"{slug}/post-history.md"] = 1

        # profile.md — LinkedIn profile rendered as markdown.
        prof = _safe(f"{slug}/profile.md", fetch_profile_md, user)
        if prof:
            (user_root / "profile.md").write_text(prof, encoding="utf-8")
            counts[f"{slug}/profile.md"] = 1

    # Shared: trigger log
    conv_dir = root / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    trigger_log = fetch_trigger_log(company_id)
    (conv_dir / "trigger-log.jsonl").write_text(trigger_log, encoding="utf-8")
    counts["conversations/trigger-log.jsonl"] = len(trigger_log.splitlines())

    return counts
