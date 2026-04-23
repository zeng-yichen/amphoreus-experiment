"""Post-hoc audit: did the Layer 1 neighbor signal actually change Stelle's behaviour?

Context
-------

``backend/src/services/post_bundle.py`` now attaches a
NEAREST CREATOR POSTS grid to each rendered block — 2 semantically-
similar priors + their reaction counts. The intent is that Stelle
sees her own content trajectory (e.g. "my last 3 posts in this
cluster landed 23/69/94 rx, declining") and calibrates without us
naming clusters or setting rules.

The concern is that we have no way to know whether this actually
works. The neighbor grid could land in the prompt and Stelle could
ignore it entirely. This module gives us a cheap, post-hoc check —
no retraining, no evaluator loop, just grep.

What it does
------------

1. Pull recent ``local_posts`` (last N days), filtered to drafts.
2. Read the ``generation_metadata.neighbor_signal_present`` flag
   that Stelle stamps at save-time (see stelle.py; stats come from
   ``post_bundle.build_post_bundle_with_stats``).
3. Scan each draft's ``content`` for markers that indicate the
   model actually reasoned about the neighbor grid:
     * ``trajectory``, ``declining``, ``stable``, ``cluster``,
       ``neighborhood`` (thematic vocabulary the neighbor framing
       invites)
     * Exact reaction-count patterns like "23 reactions" or "148 rx"
       that would only show up if the draft quoted the grid
4. Compute per-creator + aggregate rates of neighbor-influenced
   reasoning and output a report.

This is a diagnostic, not a metric. A low influence rate means one
of:
  (a) The grid isn't landing with the model (reprompt)
  (b) The markers we grep for aren't the right ones (tune markers)
  (c) The grid is influencing generation but without surface
      vocabulary (hardest to detect)
We prefer (a)/(b) — explicit reasoning is better than implicit —
so an audit showing low marker rates is a signal to tune the
bundle's preamble or the neighbor block's framing.

Usage
-----

    # library
    from backend.src.services.neighbor_signal_audit import audit
    result = audit(company="hume", days=14)

    # CLI
    python -m backend.src.services.neighbor_signal_audit hume --days 14
    python -m backend.src.services.neighbor_signal_audit --all --days 30

Never raises — returns empty stats on any upstream failure. The
audit is informational; a failure here must never block generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markers — the vocabulary we look for as proxies for "model read the grid"
# ---------------------------------------------------------------------------
#
# Keep these conservative. The grid renders as
#   NEAREST CREATOR POSTS (semantic):
#     [2026-04-16]   69 rx  "A few years ago, the biggest tech problem..."
#     [2026-04-15]   94 rx  "Most construction companies I know don't..."
#
# A draft that was meaningfully influenced by the grid will tend to
# surface either the numerical pattern (reaction counts bleeding into
# rationale), or the thematic vocabulary the grid framing invites
# (trajectory / cluster / neighborhood). Neither is proof — a post
# that happens to talk about "my last few posts went 50, 60, 70" is
# suspect too. Treat as a noisy diagnostic, not a metric.

_TRAJECTORY_WORDS = (
    "trajectory", "trending", "declining", "declined",
    "stable", "consistent",
    "cluster", "clusters", "clustering",
    "neighborhood", "neighborhoods",
    "my last three", "my last 3", "my last few", "past three posts", "past 3 posts",
)

# "23 reactions" / "23 rx" / "148 reactions · 14 comments" — reaction
# counts as numeric signals. We use this as a PROXY for the model
# quoting the grid: when neighbor data isn't in the prompt, drafts
# very rarely cite specific reaction numbers about the author's own
# past posts (there's no other source for them). With the grid in
# prompt, drafts cite them occasionally.
_REACTION_CITATION_RE = re.compile(
    r"\b(\d{1,4})\s*(reactions?|rx|likes?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public result shape
# ---------------------------------------------------------------------------

@dataclass
class AuditBucket:
    """Stats for one slice of drafts (e.g. with-neighbors, without)."""
    n_drafts:                  int = 0
    n_trajectory_words:        int = 0   # drafts mentioning any trajectory word
    n_reaction_citations:      int = 0   # drafts quoting specific reaction counts
    n_either:                  int = 0   # drafts hitting at least one marker
    examples:                  list[dict] = field(default_factory=list)  # up to 5

    @property
    def rate_trajectory(self) -> float:
        return self.n_trajectory_words / self.n_drafts if self.n_drafts else 0.0

    @property
    def rate_reaction_citation(self) -> float:
        return self.n_reaction_citations / self.n_drafts if self.n_drafts else 0.0

    @property
    def rate_either(self) -> float:
        return self.n_either / self.n_drafts if self.n_drafts else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rate_trajectory"] = round(self.rate_trajectory, 3)
        d["rate_reaction_citation"] = round(self.rate_reaction_citation, 3)
        d["rate_either"] = round(self.rate_either, 3)
        return d


@dataclass
class AuditResult:
    company:           str
    days:              int
    total_drafts:      int
    with_neighbors:    AuditBucket
    without_neighbors: AuditBucket
    unknown:           AuditBucket   # drafts with no neighbor_signal_present stamp
    skip_reason_hist:  dict[str, int] = field(default_factory=dict)
    error:             Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "company":           self.company,
            "days":              self.days,
            "total_drafts":      self.total_drafts,
            "with_neighbors":    self.with_neighbors.to_dict(),
            "without_neighbors": self.without_neighbors.to_dict(),
            "unknown":           self.unknown.to_dict(),
            "skip_reason_hist":  self.skip_reason_hist,
            "error":             self.error,
        }


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def audit(
    *, company: Optional[str] = None, days: int = 14, limit: int = 500,
) -> AuditResult:
    """Run one audit pass. ``company=None`` audits every company with drafts.

    Pulls drafts written in the last ``days``, buckets by whether the
    generation saw neighbor context (from ``generation_metadata``),
    scans each draft's body for neighbor-influenced vocabulary, and
    returns an ``AuditResult`` comparing the two buckets.

    Never raises — on any upstream failure returns an ``AuditResult``
    with ``error`` set and empty buckets.
    """
    result = AuditResult(
        company=company or "*",
        days=days,
        total_drafts=0,
        with_neighbors=AuditBucket(),
        without_neighbors=AuditBucket(),
        unknown=AuditBucket(),
    )

    try:
        from backend.src.db.amphoreus_supabase import _get_client, is_configured
    except Exception as exc:
        result.error = f"supabase import failed: {exc}"
        return result

    if not is_configured():
        result.error = "amphoreus supabase not configured (AMPHOREUS_SUPABASE_URL/KEY missing)"
        return result

    sb = _get_client()
    if sb is None:
        result.error = "amphoreus supabase client could not be constructed"
        return result

    since_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        q = (
            sb.table("local_posts")
              .select("id, company, content, created_at, generation_metadata, status")
              .gte("created_at", since_ts)
        )
        if company:
            # Match on both the slug and the resolved UUID — local_posts.company
            # can carry either depending on when the row was written. Same
            # tolerance used in post_bundle._fetch_local_posts.
            candidates = [company]
            try:
                from backend.src.lib.company_resolver import resolve_to_uuid
                resolved = resolve_to_uuid(company)
                if resolved and resolved != company:
                    candidates.append(resolved)
            except Exception:
                pass
            q = q.in_("company", candidates)
        rows = q.order("created_at", desc=True).limit(max(1, min(limit, 2000))).execute().data or []
    except Exception as exc:
        result.error = f"local_posts fetch failed: {exc}"
        return result

    for row in rows:
        content = (row.get("content") or "").strip()
        if not content:
            continue
        status = (row.get("status") or "").strip().lower()
        # Focus on drafts — published rows are already generated and
        # shipped, re-auditing them doesn't tell us about neighbor
        # influence on the current generation loop.
        if status not in ("", "draft", "rejected"):
            continue
        result.total_drafts += 1

        meta = row.get("generation_metadata") or {}
        if not isinstance(meta, dict):
            meta = {}

        skip_reason = meta.get("neighbor_skip_reason")
        if skip_reason:
            result.skip_reason_hist[skip_reason] = (
                result.skip_reason_hist.get(skip_reason, 0) + 1
            )

        if "neighbor_signal_present" in meta:
            bucket = (
                result.with_neighbors
                if meta.get("neighbor_signal_present")
                else result.without_neighbors
            )
        else:
            bucket = result.unknown

        _score_draft_into_bucket(bucket, row, content)

    return result


def _score_draft_into_bucket(
    bucket: AuditBucket, row: dict, content: str,
) -> None:
    """Scan ``content`` for neighbor-influence markers and update ``bucket``."""
    bucket.n_drafts += 1

    content_lc = content.lower()
    hit_trajectory = any(w in content_lc for w in _TRAJECTORY_WORDS)
    hit_reaction = bool(_REACTION_CITATION_RE.search(content))

    if hit_trajectory:
        bucket.n_trajectory_words += 1
    if hit_reaction:
        bucket.n_reaction_citations += 1
    if hit_trajectory or hit_reaction:
        bucket.n_either += 1
        # Keep a few concrete examples so the operator can eyeball
        # whether the markers are actually catching neighbor-influence
        # or just coincidental vocabulary. 5 is plenty.
        if len(bucket.examples) < 5:
            bucket.examples.append({
                "draft_id":    (row.get("id") or "")[:16],
                "created_at":  str(row.get("created_at") or "")[:19],
                "trajectory":  hit_trajectory,
                "reaction":    hit_reaction,
                "excerpt":     _first_matching_excerpt(content) or content[:200],
            })


def _first_matching_excerpt(content: str) -> Optional[str]:
    """Return a short excerpt around the first marker match, for the
    report's examples section. Falls back to None when nothing matches
    (caller uses head-of-content instead)."""
    content_lc = content.lower()
    for w in _TRAJECTORY_WORDS:
        idx = content_lc.find(w)
        if idx >= 0:
            start = max(0, idx - 60)
            end = min(len(content), idx + 120)
            return "…" + content[start:end].replace("\n", " ") + "…"
    m = _REACTION_CITATION_RE.search(content)
    if m:
        idx = m.start()
        start = max(0, idx - 60)
        end = min(len(content), idx + 120)
        return "…" + content[start:end].replace("\n", " ") + "…"
    return None


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _print_human(result: AuditResult) -> None:
    """Operator-friendly stdout rendering of an ``AuditResult``."""
    print()
    print(f"=== neighbor_signal_audit — company={result.company} last {result.days}d ===")
    print()
    if result.error:
        print(f"ERROR: {result.error}")
        return
    print(f"Total drafts audited: {result.total_drafts}")
    if result.skip_reason_hist:
        print()
        print("Bundle-level neighbor skip reasons:")
        for reason, ct in sorted(
            result.skip_reason_hist.items(), key=lambda t: -t[1]
        ):
            print(f"  {reason:<24} {ct}")
    print()
    for label, bucket in (
        ("WITH neighbors",    result.with_neighbors),
        ("WITHOUT neighbors", result.without_neighbors),
        ("unknown (no stamp)", result.unknown),
    ):
        print(f"--- {label}: n={bucket.n_drafts}")
        if bucket.n_drafts == 0:
            print("    (no drafts in this bucket)")
            continue
        print(
            f"    trajectory-words:   {bucket.n_trajectory_words:>3}"
            f"  ({bucket.rate_trajectory*100:5.1f}%)"
        )
        print(
            f"    reaction-citations: {bucket.n_reaction_citations:>3}"
            f"  ({bucket.rate_reaction_citation*100:5.1f}%)"
        )
        print(
            f"    either:             {bucket.n_either:>3}"
            f"  ({bucket.rate_either*100:5.1f}%)"
        )
        if bucket.examples:
            print("    examples:")
            for ex in bucket.examples:
                tags = []
                if ex["trajectory"]:
                    tags.append("traj")
                if ex["reaction"]:
                    tags.append("rx#")
                print(
                    f"      [{ex['created_at'][:10]}] {ex['draft_id']} "
                    f"({'+'.join(tags)}): {ex['excerpt']}"
                )
        print()

    # Headline comparison — the number we actually care about.
    wn, won = result.with_neighbors, result.without_neighbors
    if wn.n_drafts >= 5 and won.n_drafts >= 5:
        lift = wn.rate_either - won.rate_either
        print(
            f"Either-marker rate lift (with − without): "
            f"{lift*100:+.1f} pp  "
            f"(with={wn.rate_either*100:.1f}%, without={won.rate_either*100:.1f}%)"
        )
        if lift < 0.05:
            print(
                "  ⚠ Lift is weak. Either (a) the grid isn't landing, "
                "(b) markers need tuning, or (c) influence is implicit. "
                "See module docstring."
            )
    else:
        print(
            "Not enough data in both buckets for a lift comparison "
            "(need ≥5 drafts each)."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "company", nargs="?", default=None,
        help="Slug or UUID. Omit (or use --all) to audit every company.",
    )
    parser.add_argument("--all", action="store_true", help="Audit every company.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the human report.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load .env so AMPHOREUS_SUPABASE_URL/KEY are present when run standalone.
    try:
        from dotenv import load_dotenv
        for candidate in (
            Path(__file__).resolve().parents[3] / ".env",
            Path.cwd() / ".env",
        ):
            if candidate.exists():
                load_dotenv(candidate)
                break
    except Exception:
        pass

    company = None if args.all else args.company
    result = audit(company=company, days=args.days, limit=args.limit)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_human(result)

    return 0 if not result.error else 1


if __name__ == "__main__":
    sys.exit(main())
