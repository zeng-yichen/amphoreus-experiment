"""Posts API — CRUD, rewrite, fact-check, push to Ordinal."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/posts", tags=["posts"])


class CreatePostRequest(BaseModel):
    company: str
    content: str
    title: str | None = None
    scheduled_at: str | None = None
    status: str = "draft"


class RewriteRequest(BaseModel):
    company: str
    post_text: str
    style_instruction: str = ""
    client_context: str = ""


class FactCheckRequest(BaseModel):
    company: str
    post_text: str


class PushRequest(BaseModel):
    company: str
    content: str
    model_name: str = "stelle"
    posts_per_month: int = 12
    start_date: str | None = None


@router.get("")
async def list_posts(company: str | None = None, limit: int = 50):
    """List posts from Supabase."""
    from backend.src.db.supabase_client import get_supabase
    sb = get_supabase()
    query = sb.table("posts").select("*").order("created_at", desc=True).limit(limit)
    if company:
        query = query.eq("face_of_content_user_id", company)
    result = query.execute()
    return {"posts": result.data or []}


@router.post("")
async def create_post(req: CreatePostRequest):
    """Create a new post/draft."""
    from backend.src.db.supabase_client import get_supabase
    sb = get_supabase()
    payload = {
        "post_text": req.content,
        "hook": (req.content[:100] if req.content else ""),
        "status": req.status,
    }
    if req.scheduled_at:
        payload["post_date"] = req.scheduled_at
    result = sb.table("posts").insert(payload).execute()
    return {"post": result.data[0] if result.data else None}


@router.patch("/{post_id}")
async def update_post(post_id: str, req: CreatePostRequest):
    from backend.src.db.supabase_client import get_supabase
    sb = get_supabase()
    payload = {"post_text": req.content, "status": req.status}
    result = sb.table("posts").update(payload).eq("id", post_id).execute()
    return {"post": result.data[0] if result.data else None}


@router.delete("/{post_id}")
async def delete_post(post_id: str):
    from backend.src.db.supabase_client import get_supabase
    sb = get_supabase()
    sb.table("posts").delete().eq("id", post_id).execute()
    return {"deleted": True}


@router.post("/{post_id}/rewrite")
async def rewrite_post(post_id: str, req: RewriteRequest):
    """Rewrite a post via Cyrene."""
    from backend.src.agents.cyrene import Cyrene
    cyrene = Cyrene()
    result = cyrene.rewrite_single_post(
        post_text=req.post_text,
        style_instruction=req.style_instruction,
        client_context=req.client_context,
    )
    return {"result": result}


@router.post("/{post_id}/fact-check")
async def fact_check_post(post_id: str, req: FactCheckRequest):
    """Fact-check a post via Permansor Terrae."""
    from backend.src.agents.castorice import Castorice
    pt = Castorice()
    report = pt.fact_check_post(req.company, req.post_text)
    return {"report": report}


@router.post("/push")
async def push_to_ordinal(req: PushRequest):
    """Push posts to Ordinal via Hyacinthia."""
    from backend.src.agents.hyacinthia import Hyacinthia
    en = Hyacinthia()
    start = datetime.fromisoformat(req.start_date) if req.start_date else None
    success, result = en.push_drafts(
        company_keyword=req.company,
        model_name=req.model_name,
        content=req.content,
        posts_per_month=req.posts_per_month,
        start_date=start,
    )
    return {"success": success, "result": result}
