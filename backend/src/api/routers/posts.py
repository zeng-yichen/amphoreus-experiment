"""Posts API — CRUD, rewrite, fact-check, push to Ordinal."""

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.src.core.config import get_settings
from backend.src.lib.company_resolver import (
    resolve_to_company_and_user,
    resolve_with_fallback,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/posts", tags=["posts"])


def _canonical_company(identifier: str | None) -> str | None:
    """Normalize a caller-supplied company identifier to its canonical UUID.

    Accepts either a Jacquard ``user_companies.id`` UUID or an Amphoreus
    slug; returns the UUID whenever resolvable, or the raw identifier as a
    fallback (for legacy rows that still carry slug identifiers during the
    migration window). ``None``/empty in → ``None`` out.
    """
    if not identifier:
        return None
    return resolve_with_fallback(identifier) or identifier


def _canonical_scope(
    identifier: str | None,
) -> tuple[str | None, str | None]:
    """Return ``(company_uuid, user_id)`` for a caller-supplied identifier.

    Slugs like ``trimble-heather`` resolve to the Trimble company + Heather's
    user_id so per-FOC-user views filter correctly. Plain company slugs
    (``innovocommerce``) return ``(company_uuid, None)`` for a company-wide
    view. Falls back to the raw identifier when the resolver can't match —
    preserves legacy-row behavior during the migration window.
    """
    if not identifier:
        return None, None
    cu, uu = resolve_to_company_and_user(identifier)
    if cu is None:
        # Legacy fallback: use the raw identifier as the company filter.
        return identifier, None
    return cu, uu


def _parse_publish_at_iso(raw: str | None) -> datetime | None:
    """Parse client ISO string (often ends with Z) to timezone-aware UTC."""
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None)


def _castorice_entry_from_row(row: dict) -> dict | None:
    """Build Hyacinthia castorice_result dict from a local_posts row, or None if empty."""
    citations: list[str] = []
    cc_raw = row.get("citation_comments")
    if cc_raw:
        try:
            parsed = json.loads(cc_raw)
            if isinstance(parsed, list):
                citations = [str(x) for x in parsed]
        except json.JSONDecodeError:
            logger.warning("[posts] Invalid citation_comments JSON for post %s", row.get("id"))
    why_post = (row.get("why_post") or "").strip()
    if not citations and not why_post:
        return None
    entry: dict = {"citation_comments": citations}
    if why_post:
        entry["why_post"] = why_post
    return entry


def _utc_publish_at_nine(d: datetime) -> datetime:
    """Match push_drafts cadence: calendar day at 09:00 naive UTC for Ordinal publishAt."""
    return datetime(d.year, d.month, d.day, 9, 0, 0)


def _linkedin_asset_ids_for_row(en, company_slug: str, row: dict | None) -> list[str]:
    """Upload approved draft image to Ordinal by public URL; returns asset UUIDs for linkedIn.assetIds."""
    if not row:
        return []
    stem = (row.get("linked_image_id") or "").strip()
    if not stem:
        return []
    base = (get_settings().public_base_url or "").strip().rstrip("/")
    if not base:
        logger.warning(
            "[posts] Draft has linked_image_id=%s but PUBLIC_BASE_URL is unset — "
            "Ordinal cannot fetch the image; attach skipped.",
            stem,
        )
        return []
    public_url = f"{base}/api/images/{company_slug}/{stem}"
    aid = en.upload_asset_from_public_url(company_slug, public_url)
    return [aid] if aid else []


class CreatePostRequest(BaseModel):
    company: str
    content: str
    title: str | None = None
    scheduled_at: str | None = None
    status: str = "draft"


class PatchPostRequest(BaseModel):
    """PATCH body — use model_dump(exclude_unset=True) so omitted fields are not cleared."""

    company: str
    content: str | None = None
    title: str | None = None
    status: str | None = None
    linked_image_id: str | None = None


class RewriteRequest(BaseModel):
    company: str
    post_text: str
    style_instruction: str = ""
    client_context: str = ""


class FactCheckRequest(BaseModel):
    company: str
    post_text: str


class PushApproval(BaseModel):
    """Ordinal POST /approvals item (camelCase keys for API)."""

    userId: str
    message: str | None = None
    dueDate: str | None = None
    isBlocking: bool = False


class PushRequest(BaseModel):
    company: str
    content: str = ""
    post_id: str | None = Field(
        default=None,
        description="When set, load content, citation_comments, and why_post from this local draft row.",
    )
    model_name: str = "stelle"
    posts_per_month: int = 12
    start_date: str | None = None
    citation_comments: list[str] = []
    publish_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC instant for publishAt on Ordinal (e.g. from Date.toISOString()).",
    )
    approvals: list[PushApproval] = []


class PushAllRequest(BaseModel):
    """Push every local draft for a company to Ordinal on cadence slots (UTC)."""

    company: str
    posts_per_month: int = Field(
        12,
        description="12 → Mon/Wed/Thu slots; 8 → Tue/Thu slots (same as Hyacinthia).",
    )
    approvals: list[PushApproval] = []

    @field_validator("posts_per_month")
    @classmethod
    def posts_per_month_must_be_8_or_12(cls, v: int) -> int:
        if v not in (8, 12):
            raise ValueError("posts_per_month must be 8 or 12")
        return v


@router.get("")
async def list_posts(company: str | None = None, limit: int = 50):
    """List local draft posts for a client.

    The ``company`` param accepts either a company slug
    (``innovocommerce``), a per-FOC-user pseudo-slug (``trimble-heather``),
    or a UUID. For per-user slugs the response is filtered to that user's
    drafts only (so Heather's Posts tab doesn't show Mark's Trimble
    drafts). For company slugs it returns all company drafts.
    """
    from backend.src.db.local import list_local_posts
    company_uuid, user_id = _canonical_scope(company)
    if user_id:
        # Per-FOC-user view — pass BOTH so list_local_posts widens to
        # also include company-wide rows with no user_id stamped
        # (legacy drafts pre-user-disambiguation, or drafts where
        # submit_draft's user resolution came back None). Without this
        # the Posts tab silently hid drafts that exist in the DB.
        posts = list_local_posts(company=company_uuid, user_id=user_id, limit=limit)
    else:
        posts = list_local_posts(company=company_uuid, limit=limit)

    # Enrich paired drafts with the published post body + engagement
    # so the frontend can show the published version inline next to
    # the draft. Pair state itself is carried on the local_posts row
    # via matched_provider_urn / match_method / matched_at /
    # match_similarity (already in _LOCAL_POSTS_COLS); this just
    # joins the published body so the UI doesn't need a second fetch.
    paired_urns = [
        (p.get("matched_provider_urn") or "").strip()
        for p in posts
        if (p.get("matched_provider_urn") or "").strip()
    ]
    if paired_urns:
        try:
            from backend.src.db.amphoreus_supabase import _get_client, is_configured
            if is_configured():
                sb = _get_client()
                if sb is not None:
                    pub_rows = (
                        sb.table("linkedin_posts")
                          .select(
                              "provider_urn, post_text, posted_at, "
                              "total_reactions, total_comments, total_reposts"
                          )
                          .in_("provider_urn", paired_urns)
                          .execute()
                          .data
                        or []
                    )
                    pub_by_urn = {
                        (r.get("provider_urn") or ""): r for r in pub_rows
                    }
                    for p in posts:
                        urn = (p.get("matched_provider_urn") or "").strip()
                        if not urn:
                            continue
                        pub = pub_by_urn.get(urn)
                        if not pub:
                            continue
                        p["published_post_text"]  = pub.get("post_text") or ""
                        p["published_posted_at"]  = pub.get("posted_at") or ""
                        p["published_reactions"]  = pub.get("total_reactions") or 0
                        p["published_comments"]   = pub.get("total_comments") or 0
                        p["published_reposts"]    = pub.get("total_reposts") or 0
        except Exception as exc:
            logger.debug(
                "[posts.list] published-body enrichment failed (non-fatal): %s",
                exc,
            )

    return {"posts": posts}


@router.post("")
async def create_post(req: CreatePostRequest):
    """Create a new local draft post.

    Resolves ``req.company`` to ``(company_uuid, user_id)`` so
    per-FOC-user slugs like ``trimble-heather`` stamp both columns and
    the draft is attributable to the right person.
    """
    from backend.src.db.local import create_local_post
    post_id = str(uuid.uuid4())
    company_uuid, user_id = _canonical_scope(req.company)
    post = create_local_post(
        post_id=post_id,
        company=company_uuid or req.company,
        user_id=user_id,
        content=req.content,
        title=req.title,
        status=req.status,
    )
    return {"post": post}


@router.patch("/{post_id}")
async def update_post(post_id: str, req: PatchPostRequest):
    from backend.src.db.local import get_local_post, update_local_post_fields

    raw = req.model_dump(exclude_unset=True)
    raw.pop("company", None)
    if not raw:
        post = get_local_post(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return {"post": post}
    post = update_local_post_fields(post_id, raw)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return {"post": post}


@router.delete("/{post_id}")
async def delete_post(post_id: str, request: Request):
    """Delete a draft from the Posts tab.

    Wraps ``db.local.delete_local_post`` and surfaces any Supabase-
    side failure as a real HTTP 500 with the underlying error message
    instead of lying with ``{"deleted": true}``. Silent-success was
    the previous behaviour and produced "delete button does nothing"
    symptoms that masked real RLS / FK / permission failures.
    """
    from backend.src.db.local import delete_local_post
    user = getattr(request.state, "user", None)
    deleted_by = user.email if user and getattr(user, "email", None) else "anonymous"
    try:
        delete_local_post(post_id, deleted_by=deleted_by, reason="operator_ui")
    except Exception as exc:
        # Log at WARNING so the Fly log shows the full stack via FastAPI's
        # default exception handler; the HTTPException body shows a
        # compact version to the browser's Network tab.
        logger.warning(
            "[posts.delete_post] delete failed for %s: %s",
            post_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"delete failed: {exc}"[:400],
        )
    return {"deleted": True}


class RejectRequest(BaseModel):
    reason: str | None = Field(None, max_length=500)


@router.post("/{post_id}/reject")
async def reject_post(post_id: str, request: Request, body: RejectRequest | None = None):
    """State transition: mark a post as rejected by the client.

    Distinct from ``DELETE /{post_id}`` — preserves the ``local_posts``
    row and every paired ``draft_feedback`` row so the post stays a
    negative learning signal. Intended flow: operator sees client
    rejection on Ordinal, deletes the Ordinal draft manually, then
    hits this endpoint to preserve the draft locally with its paired
    comments. The post will surface in Stelle's / Aglaea's post bundle
    under the ``Rejected`` class (not as dedup signal — rejection
    means "this execution of this topic was wrong", not "never write
    on this topic again").
    """
    from backend.src.db.local import update_local_post_fields, get_local_post
    from backend.src.db.amphoreus_supabase import log_deletion

    existing = get_local_post(post_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Post not found")

    user = getattr(request.state, "user", None)
    rejected_by = user.email if user and getattr(user, "email", None) else "anonymous"
    reason = (body.reason if body else None) or "client rejection"

    update_local_post_fields(
        post_id,
        {"status": "rejected"},
        revision_source="reject",
        revision_author=rejected_by,
    )

    # Audit trail. Reusing deletion_log for state-transition history —
    # the entity_type discriminator makes this searchable distinct
    # from true deletions. Row is NOT deleted; the snapshot captures
    # state at the moment of rejection.
    try:
        log_deletion(
            entity_type="local_post_rejected",
            entity_id=post_id,
            entity_snapshot=existing,
            deleted_by=rejected_by,
            reason=reason,
        )
    except Exception:
        logger.debug("[posts.reject] deletion_log write failed", exc_info=True)

    return {"rejected": True, "post_id": post_id}


# ---------------------------------------------------------------------------
# Manual draft ↔ published pairing — REMOVED 2026-04-28
# ---------------------------------------------------------------------------
#
# The ``POST /api/posts/{id}/set-publish-date`` and ``DELETE`` endpoints
# plus their SetPublishDateRequest model used to let operators tell
# Amphoreus when a draft was published so it could pair to the right
# linkedin_posts row. Removed because the semantic match-back worker
# (services/draft_match_worker.run_match_back) now runs after EVERY
# Apify scrape and pairs unpaired drafts automatically — no operator
# input needed. Threshold: cosine ≥ 0.82 with margin ≥ 0.04 to second-
# best, which is strict enough that incidental topic-overlap doesnt
# trigger spurious pairings.
#
# Drafts no longer carry ``scheduled_date`` (operator never sets it)
# and ``match_method=manual_date`` is no longer written. Existing
# rows with those values keep working — readers tolerate either
# manual_date or semantic provenance.
#
# ---------------------------------------------------------------------------


@router.post("/{post_id}/rewrite")
async def rewrite_post(post_id: str, req: RewriteRequest):
    """Rewrite a post via Cyrene.

    Pulls unresolved feedback from ``draft_feedback`` and hands it to
    :meth:`Cyrene.rewrite_single_post` so the rewrite actually addresses
    what operators flagged — the highest-leverage consumption point for
    the comments system. If the feedback fetch fails (pre-migration,
    mirror unreachable, etc.) the rewrite still runs without prior
    feedback — degraded but functional.

    When the rewrite succeeds AND we have a post in ``local_posts`` for
    this id, we also persist the new content back through
    ``update_local_post`` with ``revision_source='rewrite_with_feedback'``
    so the revision history slices cleanly in eval.
    """
    prior_feedback: list[dict] = []
    try:
        from backend.src.db.amphoreus_supabase import _get_client, is_configured
        if is_configured():
            sb = _get_client()
            if sb is not None:
                prior_feedback = (
                    sb.table("draft_feedback")
                      .select("body, author_email, author_name, selected_text, source")
                      .eq("draft_id", post_id)
                      .eq("resolved", False)
                      .order("created_at", desc=False)
                      .limit(50)
                      .execute()
                      .data
                    or []
                )
    except Exception as exc:
        logger.debug("[posts/rewrite] feedback fetch failed (non-fatal): %s", exc)

    from backend.src.agents.demiurge import Cyrene
    cyrene = Cyrene()
    result = cyrene.rewrite_single_post(
        post_text=req.post_text,
        style_instruction=req.style_instruction,
        client_context=req.client_context,
        prior_feedback=prior_feedback,
    )

    # If Cyrene produced a rewrite AND this post exists in local_posts,
    # persist it so the operator's next "Rewrite" doesn't see stale
    # text, and the revision history captures the rewrite provenance.
    # The frontend may additionally PATCH the post after showing the
    # diff to the user — that's fine, the extra revision row is harmless.
    try:
        rewritten = (result or {}).get("final_post") or ""
        if rewritten.strip():
            from backend.src.db.local import get_local_post, update_local_post
            if get_local_post(post_id):
                update_local_post(
                    post_id,
                    content=rewritten,
                    revision_source="rewrite_with_feedback" if prior_feedback else "operator_edit",
                )
    except Exception:
        logger.debug("[posts/rewrite] post-update skipped", exc_info=True)

    return {"result": result, "prior_feedback_count": len(prior_feedback)}


@router.post("/{post_id}/fact-check")
async def fact_check_post(post_id: str, req: FactCheckRequest):
    """Fact-check and annotate a post via Castorice. Saves annotated version locally."""
    from backend.src.agents.castorice import Castorice
    from backend.src.db.vortex import castorice_annotated_path
    # Canonicalize so Castorice's company-keyed reads (published posts,
    # tone references, etc.) and the on-disk annotated-post mirror file
    # all use the same UUID the rest of the stack agrees on.
    company_canonical = _canonical_company(req.company) or req.company
    result = Castorice().fact_check_post(company_canonical, req.post_text)

    # Persist the annotated version for CE review
    annotated = result.get("annotated_post", "")
    if annotated:
        ann_path = castorice_annotated_path(company_canonical)
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(annotated, encoding="utf-8")
        logger.info("[posts] Annotated post saved to %s", ann_path)

    return {
        "report": result.get("report", ""),
        "corrected_post": result.get("corrected_post", ""),
        "annotated_post": annotated,
        "citation_comments": result.get("citation_comments", []),
    }


# ---------------------------------------------------------------------------
# DEPRECATED: Ordinal outbound endpoints (2026-04-23 churn begin)
# ---------------------------------------------------------------------------
#
# Virio is churning off Ordinal. Outbound endpoints (push single + push
# all + asset upload + approvals + system-comment post) are the first
# to go — they return HTTP 410 Gone with a deprecation message rather
# than executing. Everything INBOUND (analytics pulls, comment sync,
# dedup drafts query, profile resolution, key mirror) stays alive for
# now so historical engagement data keeps flowing and Stelle's dedup
# reads still see Ordinal-queued drafts during the transition.
#
# Hyacinthia's outbound functions (``push_drafts``, ``push_single_post``,
# ``upload_image_from_url``, ``create_approvals``, post-comment poster)
# still exist but are no longer reachable from the API. They can be
# deleted once the churn completes — kept for now in case an incident
# requires an emergency one-off push and we re-enable a single endpoint
# behind an env flag.
#
# Any frontend still calling these will see 410 and surface the message
# to the operator so they know push routing is gone, not broken.

_ORDINAL_OUTBOUND_DEPRECATION_MSG = (
    "Ordinal outbound is deprecated (2026-04-23). Virio is churning off "
    "Ordinal; all draft-push endpoints are disabled. Drafts continue to "
    "land in Amphoreus local_posts as usual — they just no longer route "
    "to Ordinal. Publishing will happen via the replacement pipeline "
    "once it's online. Ordinal inbound data (analytics, comments) is "
    "still mirrored into Amphoreus for historical continuity."
)


@router.post("/push")
async def push_to_ordinal(req: PushRequest):  # noqa: ARG001 — kept to preserve the 410 surface
    """Deprecated. Outbound to Ordinal was disabled 2026-04-23."""
    raise HTTPException(status_code=410, detail=_ORDINAL_OUTBOUND_DEPRECATION_MSG)


@router.post("/push-all")
async def push_all_drafts_to_ordinal(req: PushAllRequest):  # noqa: ARG001
    """Deprecated. Outbound to Ordinal was disabled 2026-04-23."""
    raise HTTPException(status_code=410, detail=_ORDINAL_OUTBOUND_DEPRECATION_MSG)


# ---------------------------------------------------------------------------
# Feedback (draft_feedback) + revisions (local_post_revisions)
#
# Content engineers reviewing Stelle's output leave notes on drafts here.
# Feedback rows are consumed at rewrite-time by Cyrene so the rewrite
# actually addresses what operators flagged. Revisions are an audit
# trail of every content change — needed for the revert button and for
# future diff views. See
# backend/scripts/amphoreus_supabase_feedback_schema.sql for the table
# shape, and db/local.py::_record_content_revision for the write hook.
# ---------------------------------------------------------------------------


class CommentRequest(BaseModel):
    """Create a feedback row on a draft.

    ``selection_*`` are either all present (inline comment on a
    highlighted range) or all absent (post-wide comment). The frontend
    enforces this; the backend accepts both shapes.
    """
    body: str = Field(..., min_length=1, max_length=5000)
    source: str = Field(
        default="operator_postwide",
        pattern=r"^(operator_postwide|operator_inline)$",
    )
    author_email: str | None = None
    author_name: str | None = None
    selection_start: int | None = None
    selection_end: int | None = None
    selected_text: str | None = None


def _feedback_sb():
    """Return the Amphoreus Supabase client, or raise 503.

    Central so all feedback endpoints fail the same way when the
    mirror isn't configured — avoids partial-outcome surprises where
    one endpoint works and another silently does nothing.
    """
    from backend.src.db.amphoreus_supabase import _get_client, is_configured
    if not is_configured():
        raise HTTPException(status_code=503, detail="Amphoreus Supabase not configured")
    sb = _get_client()
    if sb is None:
        raise HTTPException(status_code=503, detail="Amphoreus Supabase client unavailable")
    return sb


@router.get("/{post_id}/comments")
async def list_comments(post_id: str, include_resolved: bool = True):
    """List feedback on a draft, newest first.

    Set ``include_resolved=false`` to show only open comments — useful
    when rendering a compact "things to address" indicator next to the
    post.
    """
    sb = _feedback_sb()
    try:
        q = sb.table("draft_feedback").select("*").eq("draft_id", post_id)
        if not include_resolved:
            q = q.eq("resolved", False)
        rows = q.order("created_at", desc=True).limit(500).execute().data or []
    except Exception as exc:
        # Missing table (pre-migration) → empty list, don't 500.
        logger.debug("[posts/comments] list failed: %s", exc)
        rows = []
    return {"comments": rows}


@router.post("/{post_id}/comments")
async def add_comment(post_id: str, req: CommentRequest):
    """Add a feedback row. Returns the persisted row."""
    # Inline comments need all three selection fields or none — otherwise
    # rendering can't anchor the comment. Fail fast rather than store
    # junk.
    has_start = req.selection_start is not None
    has_end = req.selection_end is not None
    has_text = bool(req.selected_text)
    if req.source == "operator_inline":
        if not (has_start and has_end and has_text):
            raise HTTPException(
                status_code=400,
                detail="operator_inline comments require selection_start, selection_end, and selected_text",
            )
        if req.selection_end <= req.selection_start:
            raise HTTPException(
                status_code=400,
                detail="selection_end must be > selection_start",
            )
    else:
        # Post-wide: normalise to null so the resolved schema constraints
        # stay clean regardless of what the UI posted.
        if has_start or has_end or has_text:
            logger.debug("[posts/comments] ignoring selection fields on post-wide comment")

    sb = _feedback_sb()
    payload = {
        "draft_id": post_id,
        "source": req.source,
        "body": req.body.strip(),
        "author_email": req.author_email,
        "author_name": req.author_name,
        "selection_start": req.selection_start if req.source == "operator_inline" else None,
        "selection_end":   req.selection_end   if req.source == "operator_inline" else None,
        "selected_text":   req.selected_text   if req.source == "operator_inline" else None,
    }
    try:
        resp = sb.table("draft_feedback").insert(payload).execute()
    except Exception as exc:
        logger.exception("[posts/comments] insert failed")
        raise HTTPException(status_code=500, detail=f"insert failed: {exc}")
    row = (resp.data or [{}])[0]
    return {"comment": row}


class EditCommentRequest(BaseModel):
    """Edit the body of an existing comment in place."""
    body: str = Field(..., min_length=1, max_length=5000)


@router.patch("/{post_id}/comments/{comment_id}")
async def edit_comment(post_id: str, comment_id: str, req: EditCommentRequest):
    """Edit a comment's body. Selection anchors and source stay
    immutable — if you wanted to anchor differently, delete + recreate.

    Editing an Ordinal-sourced comment is allowed but a little weird —
    the upstream Ordinal row stays unchanged, and the next sync pass
    won't overwrite us (we only insert *new* external_ids). Treat edits
    on ordinal-sourced rows as local annotations.
    """
    sb = _feedback_sb()
    try:
        sb.table("draft_feedback").update({
            "body": req.body.strip(),
        }).eq("id", comment_id).eq("draft_id", post_id).execute()
    except Exception as exc:
        logger.exception("[posts/comments] edit failed")
        raise HTTPException(status_code=500, detail=f"edit failed: {exc}")
    return {"updated": True, "id": comment_id}


@router.post("/{post_id}/comments/{comment_id}/resolve")
async def resolve_comment(post_id: str, comment_id: str, resolved_by: str | None = None):
    """Mark a comment resolved. Idempotent — resolving an already-resolved
    row is a no-op and returns the existing state."""
    sb = _feedback_sb()
    try:
        sb.table("draft_feedback").update({
            "resolved": True,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "resolved_by": resolved_by,
        }).eq("id", comment_id).eq("draft_id", post_id).execute()
    except Exception as exc:
        logger.exception("[posts/comments] resolve failed")
        raise HTTPException(status_code=500, detail=f"resolve failed: {exc}")
    return {"resolved": True, "id": comment_id}


@router.delete("/{post_id}/comments/{comment_id}")
async def delete_comment(post_id: str, comment_id: str):
    """Hard-delete a comment. Used for accidental posts — prefer
    ``resolve`` for the normal 'handled, keep audit trail' case."""
    sb = _feedback_sb()
    try:
        sb.table("draft_feedback").delete().eq("id", comment_id).eq("draft_id", post_id).execute()
    except Exception as exc:
        logger.exception("[posts/comments] delete failed")
        raise HTTPException(status_code=500, detail=f"delete failed: {exc}")
    return {"deleted": True, "id": comment_id}


@router.post("/{post_id}/revert")
async def revert_to_original(post_id: str):
    """Restore ``local_posts.content`` to the Stelle-initial / pre-Castorice
    version stored in ``pre_revision_content``.

    No-op (with a 409) if the draft has no pre-revision content on file —
    that means it was written before we started capturing originals and
    we'd be reverting to nothing.
    """
    from backend.src.db.local import get_local_post, update_local_post
    row = get_local_post(post_id)
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")
    original = (row.get("pre_revision_content") or "").strip()
    if not original:
        raise HTTPException(
            status_code=409,
            detail="no pre-revision content on file — this draft has no captured original to revert to.",
        )
    updated = update_local_post(
        post_id,
        content=original,
        revision_source="revert_to_original",
    )
    return {"reverted": True, "post": updated}


@router.get("/{post_id}/revisions")
async def list_revisions(post_id: str, limit: int = 50):
    """Return the content-revision history for a draft, newest first.
    The MVP UI only surfaces this for the revert use-case; a diff
    viewer can come later without another schema change."""
    sb = _feedback_sb()
    try:
        rows = (
            sb.table("local_post_revisions")
              .select("id, draft_id, source, author_email, created_at, content")
              .eq("draft_id", post_id)
              .order("created_at", desc=True)
              .limit(max(1, min(limit, 500)))
              .execute()
              .data
            or []
        )
    except Exception as exc:
        logger.debug("[posts/revisions] list failed: %s", exc)
        rows = []
    return {"revisions": rows}
