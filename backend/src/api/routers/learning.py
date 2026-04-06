"""Learning Intelligence API — serves client learning dashboard data."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import APIRouter

from backend.src.db import vortex as P

router = APIRouter(prefix="/api/learning", tags=["learning"])


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sparkline(values: list[float], width: int = 12) -> str:
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    vals = values[-width:]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    return "".join(bars[min(len(bars) - 1, int((v - mn) / rng * (len(bars) - 1)))] for v in vals)


def _collect_client(company: str) -> dict:
    """Collect all learning data for a single client."""
    data: dict = {"company": company}

    # --- RuanMei ---
    # Use the SQLite load path (same as RuanMei, ordinal_sync, and all other
    # consumers). The old JSON file at ruan_mei_state.json may be stale —
    # after the SQLite migration, saves go to SQLite and the JSON isn't updated.
    rm_state = None
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        rm_state = ruan_mei_load(company)
    except Exception:
        rm_state = _load_json(P.memory_dir(company) / "ruan_mei_state.json")
    obs = rm_state.get("observations", []) if rm_state else []
    scored = [o for o in obs if o.get("status") == "scored"]
    pending = [o for o in obs if o.get("status") == "pending"]

    data["observations"] = {
        "total": len(obs), "scored": len(scored), "pending": len(pending),
        "pct_scored": round(len(scored) / max(len(obs), 1) * 100, 1),
    }

    rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
    if rewards:
        rewards_sorted = sorted(rewards)
        n = len(rewards)
        data["reward_stats"] = {
            "min": round(min(rewards), 3), "max": round(max(rewards), 3),
            "mean": round(sum(rewards) / n, 3),
            "median": round(rewards_sorted[n // 2], 3),
            "std": round(math.sqrt(sum((r - sum(rewards)/n)**2 for r in rewards) / n), 3),
        }
    else:
        data["reward_stats"] = {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}

    data["reward_sparkline"] = _sparkline(rewards)

    # Engagement averages
    n_s = max(len(scored), 1)
    data["engagement"] = {
        "avg_impressions": round(sum(o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0) for o in scored) / n_s, 1),
        "avg_reactions": round(sum(o.get("reward", {}).get("raw_metrics", {}).get("reactions", 0) for o in scored) / n_s, 1),
        "avg_comments": round(sum(o.get("reward", {}).get("raw_metrics", {}).get("comments", 0) for o in scored) / n_s, 1),
    }

    # Cadence
    timestamps = []
    for o in scored:
        ts = o.get("posted_at", "")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except Exception:
                pass
    timestamps.sort()
    gaps = [(timestamps[i] - timestamps[i-1]).total_seconds() / 86400
            for i in range(1, len(timestamps)) if (timestamps[i] - timestamps[i-1]).total_seconds() > 0]
    data["cadence"] = {
        "avg_days": round(sum(gaps) / max(len(gaps), 1), 1) if gaps else 0,
        "posts_last_7d": sum(1 for t in timestamps if (datetime.now(timezone.utc) - t).days <= 7),
    }

    # --- LOLA ---
    lola = _load_json(P.memory_dir(company) / "lola_state.json")
    if lola:
        arms = lola.get("arms", [])
        points = lola.get("points", [])
        sorted_arms = sorted([a for a in arms if a.get("n_pulls", 0) > 0],
                             key=lambda a: a.get("sum_reward", 0) / max(a.get("n_pulls", 1), 1), reverse=True)
        data["lola"] = {
            "total_pulls": lola.get("total_pulls", 0),
            "arm_count": len(arms),
            "content_points": len(points),
            "using_continuous": len(points) >= 10,
            "exploration_rate": lola.get("thresholds", {}).get("exploration_rate", 0.2),
            "top_arms": [{"label": a.get("label", ""), "pulls": a.get("n_pulls", 0),
                          "avg_reward": round(a.get("sum_reward", 0) / max(a.get("n_pulls", 1), 1), 3)}
                         for a in sorted_arms[:5]],
            "bottom_arms": [{"label": a.get("label", ""), "pulls": a.get("n_pulls", 0),
                             "avg_reward": round(a.get("sum_reward", 0) / max(a.get("n_pulls", 1), 1), 3)}
                            for a in sorted_arms[-3:]] if len(sorted_arms) >= 3 else [],
        }
    else:
        data["lola"] = None

    # --- Observation tags ---
    from collections import Counter
    topic_dist = Counter(o.get("topic_tag") for o in scored if o.get("topic_tag"))
    format_dist = Counter(o.get("format_tag") for o in scored if o.get("format_tag"))
    data["tags"] = {
        "tagged_count": sum(1 for o in scored if o.get("topic_tag")),
        "topics": dict(topic_dist.most_common()),
        "formats": dict(format_dist.most_common()),
    }

    # --- Analyst findings ---
    analyst = _load_json(P.memory_dir(company) / "analyst_findings.json")
    if analyst:
        findings = analyst.get("findings", [])
        runs = analyst.get("runs", [])
        last_run = runs[-1] if runs else {}
        latest_run_id = last_run.get("run_id", "")

        # Only surface latest run's findings
        if latest_run_id:
            latest_findings = [f for f in findings if f.get("run_id") == latest_run_id]
        else:
            latest_findings = findings[-10:]

        data["analyst"] = {
            "total_runs": len(runs),
            "total_findings_all_time": len(findings),
            "latest_findings": [
                {
                    "claim": f.get("claim", ""),
                    "confidence": f.get("confidence", "?"),
                    "evidence": f.get("evidence", "")[:200],
                }
                for f in latest_findings
            ],
            "last_run": {
                "timestamp": last_run.get("timestamp", ""),
                "tool_calls": last_run.get("tool_calls", 0),
                "findings_stored": last_run.get("findings_stored", 0),
                "elapsed_seconds": last_run.get("elapsed_seconds", 0),
            } if last_run else None,
        }
    else:
        data["analyst"] = None

    # --- Learned directives ---
    directives = _load_json(P.memory_dir(company) / "learned_directives.json")
    if directives:
        data["directives"] = {
            "count": len(directives.get("directives", [])),
            "rules": [
                {
                    "directive": d.get("directive", ""),
                    "priority": d.get("priority", "medium"),
                    "source": d.get("source", "?"),
                    "efficacy": d.get("efficacy_classification", "untested"),
                }
                for d in directives.get("directives", [])
            ],
        }
    else:
        data["directives"] = None

    # --- Adaptive readiness ---
    obs_with_perm = sum(1 for o in obs if o.get("cyrene_dimensions"))
    data["readiness"] = {
        "cyrene_dims": obs_with_perm,
        "cyrene_weights_ready": obs_with_perm >= 10,
        "freeform_critic_active": len(scored) >= 10,
        "observation_tagger_active": data["tags"]["tagged_count"] > 0,
        "analyst_active": data["analyst"] is not None and bool(data["analyst"].get("latest_findings")),
        "directives_active": data["directives"] is not None and data["directives"]["count"] > 0,
    }

    # --- ICP ---
    icp = _load_json(P.icp_definition_path(company))
    data["icp"] = icp

    return data


@router.get("/clients")
async def list_learning_clients():
    """List all clients with learning data."""
    clients = []
    if P.MEMORY_ROOT.exists():
        for d in sorted(P.MEMORY_ROOT.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name != "our_memory":
                rm = _load_json(d / "ruan_mei_state.json")
                if rm:
                    obs = rm.get("observations", [])
                    scored = sum(1 for o in obs if o.get("status") == "scored")
                    clients.append({"slug": d.name, "scored": scored, "total": len(obs)})
    return {"clients": clients}


@router.get("/clients/{company}")
async def get_learning_detail(company: str):
    """Get full learning dashboard data for a client."""
    return _collect_client(company)


@router.get("/cross-client")
async def get_cross_client():
    """Get cross-client learning summary."""
    hook_lib = _load_json(P.our_memory_dir() / "hook_library.json")
    patterns = _load_json(P.our_memory_dir() / "universal_patterns.json")

    # Collect top/bottom arms across all clients
    all_arms = []
    if P.MEMORY_ROOT.exists():
        for d in P.MEMORY_ROOT.iterdir():
            if not d.is_dir() or d.name.startswith(".") or d.name == "our_memory":
                continue
            lola = _load_json(d / "lola_state.json")
            if not lola:
                continue
            for a in lola.get("arms", []):
                if a.get("n_pulls", 0) > 0:
                    all_arms.append({
                        "company": d.name,
                        "label": a.get("label", ""),
                        "pulls": a.get("n_pulls", 0),
                        "avg_reward": round(a.get("sum_reward", 0) / max(a.get("n_pulls", 1), 1), 3),
                    })

    all_arms.sort(key=lambda a: a["avg_reward"], reverse=True)

    return {
        "hook_library_size": len(hook_lib) if isinstance(hook_lib, list) else 0,
        "universal_patterns": len(patterns) if isinstance(patterns, list) else 0,
        "patterns": patterns[:5] if isinstance(patterns, list) else [],
        "top_arms": all_arms[:5],
        "bottom_arms": all_arms[-5:] if len(all_arms) >= 5 else [],
    }
