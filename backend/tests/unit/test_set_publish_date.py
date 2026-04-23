"""Unit tests for POST /api/posts/{id}/set-publish-date.

Covers the six outcomes of the endpoint:

  * happy path — single LinkedIn post on the given LA day → paired
  * 404 — draft id doesn't exist
  * 409 — zero LinkedIn posts on that day (not scraped yet)
  * 409 — multiple LinkedIn posts on that day (ambiguous)
  * 400 — malformed publish_date
  * 409 — draft has no resolvable company

…plus one timezone-correctness test confirming the PT-calendar-day
window hits the expected UTC range across a DST-boundary date.

Run:
    pytest backend/tests/unit/test_set_publish_date.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.src.api.routers.posts import router as posts_router


# ---------------------------------------------------------------------------
# FastAPI harness — a minimal app wrapping just the posts router, no CF
# Access verifier or other middleware. Keeps the test surface tight.
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(posts_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mock helpers — every set_publish_date call hits four dependencies:
#   * get_local_post (db.local)
#   * update_local_post_fields (db.local)
#   * _resolve_linkedin_username (agents.stelle)
#   * _get_client / is_configured (db.amphoreus_supabase)
# The Supabase client is shaped as a chainable mock: .table().select().eq()...
# .execute() returns an object with .data list.
# ---------------------------------------------------------------------------

class _SBResp:
    def __init__(self, data):
        self.data = data


class _SBChain:
    """Chainable supabase-py stub — every filter/sort method returns self
    so the query builder can be written fluently in the handler. The
    terminal ``.execute()`` call returns the pre-seeded data list."""
    def __init__(self, rows):
        self._rows = rows

    def select(self, *a, **kw): return self
    def eq(self, *a, **kw):     return self
    def gte(self, *a, **kw):    return self
    def lte(self, *a, **kw):    return self
    def order(self, *a, **kw):  return self
    def limit(self, *a, **kw):  return self
    def execute(self):          return _SBResp(self._rows)


def _mock_sb(linkedin_rows):
    """Build a supabase-py mock that returns ``linkedin_rows`` for any
    ``.table('linkedin_posts').select(...)`` query."""
    sb = MagicMock()
    sb.table.return_value = _SBChain(linkedin_rows)
    return sb


def _patch_stack(
    *,
    draft_row,
    linkedin_rows,
    resolved_username="sachilv",
    is_configured=True,
):
    """Install the four patches shared by every endpoint test. Returns
    the list of context managers so the test can enter them in one with."""
    return [
        patch(
            "backend.src.db.local.get_local_post",
            return_value=draft_row,
        ),
        patch(
            "backend.src.db.local.update_local_post_fields",
            return_value=None,
        ),
        patch(
            "backend.src.agents.stelle._resolve_linkedin_username",
            return_value=resolved_username,
        ),
        patch(
            "backend.src.db.amphoreus_supabase.is_configured",
            return_value=is_configured,
        ),
        patch(
            "backend.src.db.amphoreus_supabase._get_client",
            return_value=_mock_sb(linkedin_rows),
        ),
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_single_match(client):
    draft = {
        "id":      "draft-1",
        "company": "co-uuid-1",
        "user_id": "user-1",
        "status":  "draft",
    }
    li_row = {
        "provider_urn":   "7450000000000000000",
        "post_text":      "Published hook — first line\nBody follows.",
        "posted_at":      "2026-04-23T16:00:00+00:00",
        "total_reactions": 9,
        "total_comments":  1,
        "total_reposts":   0,
        "_source":         "amphoreus",
    }
    with patch("backend.src.db.local.get_local_post", return_value=draft), \
         patch("backend.src.db.local.update_local_post_fields") as upd, \
         patch("backend.src.agents.stelle._resolve_linkedin_username", return_value="sachilv"), \
         patch("backend.src.db.amphoreus_supabase.is_configured", return_value=True), \
         patch("backend.src.db.amphoreus_supabase._get_client", return_value=_mock_sb([li_row])):
        r = client.post("/api/posts/draft-1/set-publish-date",
                        json={"publish_date": "2026-04-23"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["paired"] is True
    assert body["matched_provider_urn"] == "7450000000000000000"
    assert body["matched_reactions"] == 9
    assert body["matched_hook"].startswith("Published hook")

    # Verify the update stamped the right fields on the right row.
    upd.assert_called_once()
    (post_id, fields), _ = upd.call_args
    assert post_id == "draft-1"
    assert fields["matched_provider_urn"] == "7450000000000000000"
    assert fields["match_method"]         == "manual_date"
    assert fields["match_similarity"]     is None  # not applicable for manual
    assert "matched_at" in fields


# ---------------------------------------------------------------------------
# 404 — draft doesn't exist
# ---------------------------------------------------------------------------

def test_returns_404_when_draft_missing(client):
    with patch("backend.src.db.local.get_local_post", return_value=None):
        r = client.post("/api/posts/missing/set-publish-date",
                        json={"publish_date": "2026-04-23"})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 409 — zero matches on that LA day
# ---------------------------------------------------------------------------

def test_returns_409_when_no_linkedin_post_on_date(client):
    draft = {"id": "d", "company": "co", "user_id": "u", "status": "draft"}
    with patch("backend.src.db.local.get_local_post", return_value=draft), \
         patch("backend.src.db.local.update_local_post_fields") as upd, \
         patch("backend.src.agents.stelle._resolve_linkedin_username", return_value="sachilv"), \
         patch("backend.src.db.amphoreus_supabase.is_configured", return_value=True), \
         patch("backend.src.db.amphoreus_supabase._get_client", return_value=_mock_sb([])):
        r = client.post("/api/posts/d/set-publish-date",
                        json={"publish_date": "2026-04-23"})
    assert r.status_code == 409
    assert "sachilv" in r.json()["detail"]
    assert "2026-04-23" in r.json()["detail"]
    # No pairing should have been recorded.
    upd.assert_not_called()


# ---------------------------------------------------------------------------
# 409 — multiple matches (ambiguous date)
# ---------------------------------------------------------------------------

def test_returns_409_when_multiple_posts_on_date(client):
    draft = {"id": "d", "company": "co", "user_id": "u", "status": "draft"}
    li_rows = [
        {"provider_urn": "u1", "post_text": "hook 1", "posted_at": "2026-04-23T09:00:00Z",
         "total_reactions": 3, "total_comments": 0, "total_reposts": 0},
        {"provider_urn": "u2", "post_text": "hook 2", "posted_at": "2026-04-23T18:00:00Z",
         "total_reactions": 5, "total_comments": 1, "total_reposts": 0},
    ]
    with patch("backend.src.db.local.get_local_post", return_value=draft), \
         patch("backend.src.db.local.update_local_post_fields") as upd, \
         patch("backend.src.agents.stelle._resolve_linkedin_username", return_value="sachilv"), \
         patch("backend.src.db.amphoreus_supabase.is_configured", return_value=True), \
         patch("backend.src.db.amphoreus_supabase._get_client", return_value=_mock_sb(li_rows)):
        r = client.post("/api/posts/d/set-publish-date",
                        json={"publish_date": "2026-04-23"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    # detail is a structured dict when ambiguous (vs a plain string for zero)
    assert isinstance(detail, dict)
    assert detail["error"] == "ambiguous_date"
    assert len(detail["candidates"]) == 2
    assert {c["provider_urn"] for c in detail["candidates"]} == {"u1", "u2"}
    upd.assert_not_called()


# ---------------------------------------------------------------------------
# 400 — malformed date
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_date", [
    "",
    "2026/04/23",
    "April 23",
    "2026-4-23",      # month not zero-padded, violates pattern
    "23-04-2026",
    "2026-13-40",     # correct shape, invalid month/day — 400 from ctor
])
def test_rejects_malformed_date(client, bad_date):
    # Note: the pattern-valid-but-date-invalid case (2026-13-40) must reach
    # the handler to trigger the 400 from `datetime(y, m, d, ...)`. The
    # pattern-invalid cases are rejected by pydantic before the handler,
    # returning 422 — check both paths.
    draft = {"id": "d", "company": "co", "user_id": "u", "status": "draft"}
    with patch("backend.src.db.local.get_local_post", return_value=draft), \
         patch("backend.src.agents.stelle._resolve_linkedin_username", return_value="sachilv"):
        r = client.post("/api/posts/d/set-publish-date",
                        json={"publish_date": bad_date})
    assert r.status_code in (400, 422), f"got {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# 409 — draft has no company
# ---------------------------------------------------------------------------

def test_returns_400_when_draft_has_no_company(client):
    draft = {"id": "d", "company": "", "user_id": "u", "status": "draft"}
    with patch("backend.src.db.local.get_local_post", return_value=draft):
        r = client.post("/api/posts/d/set-publish-date",
                        json={"publish_date": "2026-04-23"})
    assert r.status_code == 400
    assert "company" in r.json()["detail"].lower()


def test_returns_409_when_username_unresolvable(client):
    draft = {"id": "d", "company": "co", "user_id": "u", "status": "draft"}
    with patch("backend.src.db.local.get_local_post", return_value=draft), \
         patch("backend.src.agents.stelle._resolve_linkedin_username", return_value=None):
        r = client.post("/api/posts/d/set-publish-date",
                        json={"publish_date": "2026-04-23"})
    assert r.status_code == 409
    assert "linkedin" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Timezone correctness
# ---------------------------------------------------------------------------
#
# The endpoint interprets ``publish_date`` as a calendar day in
# America/Los_Angeles and converts to a UTC instant range. We can't
# intercept PostgREST params directly through the chainable mock, but
# we CAN sanity-check the resolved range by stashing the .gte() and
# .lte() arguments on a spy.
#
# The LA-timezone conversion matters because during DST (PDT, UTC-7)
# the window is 07:00 UTC → 07:00 next-day UTC, vs standard time
# (PST, UTC-8) where it's 08:00 → 08:00. Late-April 2026 is inside
# PDT, so we expect 07:00 UTC.

class _CaptureChain(_SBChain):
    """_SBChain variant that records the posted_at gte/lte args on
    ``self.captured`` for the timezone assertion."""
    def __init__(self, rows):
        super().__init__(rows)
        self.captured: dict[str, object] = {}

    def gte(self, col, value):
        if col == "posted_at":
            self.captured["gte"] = value
        return self

    def lte(self, col, value):
        if col == "posted_at":
            self.captured["lte"] = value
        return self


def test_timezone_range_is_pacific_local_day(client):
    draft = {"id": "d", "company": "co", "user_id": "u", "status": "draft"}
    cap = _CaptureChain([])  # no LI rows; we're after the query bounds
    sb = MagicMock()
    sb.table.return_value = cap

    with patch("backend.src.db.local.get_local_post", return_value=draft), \
         patch("backend.src.agents.stelle._resolve_linkedin_username", return_value="sachilv"), \
         patch("backend.src.db.amphoreus_supabase.is_configured", return_value=True), \
         patch("backend.src.db.amphoreus_supabase._get_client", return_value=sb):
        r = client.post("/api/posts/d/set-publish-date",
                        json={"publish_date": "2026-04-23"})
    # 409 (zero matches) is expected here; we only care about the
    # captured bounds.
    assert r.status_code == 409

    gte = str(cap.captured.get("gte", ""))
    lte = str(cap.captured.get("lte", ""))

    # 2026-04-23 is inside PDT (UTC-7). 00:00 LA → 07:00 UTC same day.
    # Accept either "+00:00" or "Z" tz suffix; Python's isoformat
    # emits +00:00 by default.
    assert "2026-04-23T07:00:00" in gte, f"gte bound wrong: {gte}"
    # End-of-day 23:59:59.999999 LA (PDT) → 06:59:59.999999 UTC next day.
    assert "2026-04-24T06:59:59" in lte, f"lte bound wrong: {lte}"


# ---------------------------------------------------------------------------
# Unset / unpair
# ---------------------------------------------------------------------------

def test_unset_clears_match_fields(client):
    draft = {"id": "d", "company": "co", "user_id": "u",
             "matched_provider_urn": "old-urn", "match_method": "manual_date"}
    with patch("backend.src.db.local.get_local_post", return_value=draft), \
         patch("backend.src.db.local.update_local_post_fields") as upd:
        r = client.delete("/api/posts/d/set-publish-date")
    assert r.status_code == 200
    assert r.json() == {"unpaired": True, "post_id": "d"}
    (post_id, fields), _ = upd.call_args
    assert post_id == "d"
    assert fields == {
        "matched_provider_urn": None,
        "matched_at":           None,
        "match_similarity":     None,
        "match_method":         None,
    }


def test_unset_returns_404_when_missing(client):
    with patch("backend.src.db.local.get_local_post", return_value=None):
        r = client.delete("/api/posts/nope/set-publish-date")
    assert r.status_code == 404
