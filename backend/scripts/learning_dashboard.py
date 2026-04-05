#!/usr/bin/env python3
"""Amphoreus Learning Dashboard — reads all client state files and prints
a comprehensive learning progress report.

Usage:
    python3 backend/scripts/learning_dashboard.py --client hensley-biostats
    python3 backend/scripts/learning_dashboard.py --all
    python3 backend/scripts/learning_dashboard.py --all --json > /tmp/dashboard.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from backend.src.db import vortex as P


# ------------------------------------------------------------------
# Data collection helpers
# ------------------------------------------------------------------

def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sparkline(values: list[float], width: int = 8) -> str:
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    vals = values[-width:]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    return "".join(bars[min(len(bars) - 1, int((v - mn) / rng * (len(bars) - 1)))] for v in vals)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    n = len(values)
    mn = min(values)
    mx = max(values)
    mean = sum(values) / n
    s = sorted(values)
    median = s[n // 2]
    var = sum((v - mean) ** 2 for v in values) / max(n, 1)
    std = math.sqrt(var)
    return {"min": round(mn, 3), "max": round(mx, 3), "mean": round(mean, 3),
            "median": round(median, 3), "std": round(std, 3)}


def _fmt_num(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.0f}"


# ------------------------------------------------------------------
# Per-client data
# ------------------------------------------------------------------

def collect_client(company: str) -> dict:
    data: dict = {"company": company}

    # --- RuanMei ---
    rm_state = _load_json(P.ruan_mei_state_path(company))
    if rm_state is None:
        rm_state = _load_json(P.memory_dir(company) / "ruan_mei_state.json")
    obs = rm_state.get("observations", []) if rm_state else []
    scored = [o for o in obs if o.get("status") == "scored"]
    pending = [o for o in obs if o.get("status") == "pending"]

    data["observations"] = {
        "total": len(obs),
        "scored": len(scored),
        "pending": len(pending),
        "pct_scored": round(len(scored) / max(len(obs), 1) * 100, 1),
    }

    rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
    data["reward_stats"] = _stats(rewards)
    data["reward_sparkline"] = _sparkline(rewards)

    # Raw engagement averages
    impressions = [o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0) for o in scored]
    reactions = [o.get("reward", {}).get("raw_metrics", {}).get("reactions", 0) for o in scored]
    comments = [o.get("reward", {}).get("raw_metrics", {}).get("comments", 0) for o in scored]
    reposts = [o.get("reward", {}).get("raw_metrics", {}).get("reposts", 0) for o in scored]
    n_s = max(len(scored), 1)
    data["engagement"] = {
        "avg_impressions": round(sum(impressions) / n_s, 1),
        "avg_reactions": round(sum(reactions) / n_s, 1),
        "avg_comments": round(sum(comments) / n_s, 1),
        "avg_reposts": round(sum(reposts) / n_s, 1),
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
    lola_state = _load_json(P.memory_dir(company) / "lola_state.json")
    if lola_state:
        arms = lola_state.get("arms", [])
        active = [a for a in arms if not a.get("retired")]
        retired = [a for a in arms if a.get("retired") and not a.get("icp_retired")]
        icp_retired = [a for a in arms if a.get("icp_retired")]
        icp_boosted = [a for a in arms if a.get("icp_boosted")]
        resting = [a for a in arms if a.get("retired") and a.get("rest_counter", 0) > 0]

        sorted_arms = sorted([a for a in arms if a.get("n_pulls", 0) > 0],
                             key=lambda a: a.get("sum_reward", 0) / max(a.get("n_pulls", 1), 1),
                             reverse=True)
        top3 = sorted_arms[:3]
        bottom3 = sorted_arms[-3:] if len(sorted_arms) >= 3 else []

        thresholds = lola_state.get("thresholds", {})

        data["lola"] = {
            "total_pulls": lola_state.get("total_pulls", 0),
            "arm_count": len(arms),
            "active": len(active),
            "retired": len(retired),
            "icp_retired": len(icp_retired),
            "icp_boosted": len(icp_boosted),
            "resting": len(resting),
            "exploration_rate": thresholds.get("exploration_rate", 0.2),
            "top_arms": [{
                "label": a.get("label", ""),
                "pulls": a.get("n_pulls", 0),
                "avg_reward": round(a.get("sum_reward", 0) / max(a.get("n_pulls", 1), 1), 3),
            } for a in top3],
            "bottom_arms": [{
                "label": a.get("label", ""),
                "pulls": a.get("n_pulls", 0),
                "avg_reward": round(a.get("sum_reward", 0) / max(a.get("n_pulls", 1), 1), 3),
            } for a in bottom3],
        }
    else:
        data["lola"] = None

    # --- Adaptive config ---
    ac = _load_json(P.memory_dir(company) / "adaptive_config.json")
    data["adaptive"] = {}
    if ac:
        for module in ("permansor", "constitutional", "temporal", "feedback"):
            entry = ac.get(module)
            if entry:
                data["adaptive"][module] = {
                    "tier": entry.get("_tier", "unknown"),
                    "computed_at": entry.get("_computed_at", ""),
                }
                if module == "permansor":
                    data["adaptive"][module]["pass_threshold"] = entry.get("pass_threshold")
                    data["adaptive"][module]["dimension_weights"] = entry.get("dimension_weights")
                elif module == "constitutional":
                    data["adaptive"][module]["soft_principles"] = entry.get("soft_principles", [])

    # --- Adaptive readiness ---
    obs_with_perm = sum(1 for o in obs if o.get("permansor_dimensions"))
    obs_with_const = sum(1 for o in obs if o.get("constitutional_results"))
    data["readiness"] = {
        "permansor_dims_collected": obs_with_perm,
        "permansor_weights_need": max(0, 10 - obs_with_perm),
        "constitutional_collected": obs_with_const,
        "constitutional_soft_need": max(0, 15 - obs_with_const),
        "emergent_dims_need": max(0, 40 - obs_with_perm),
        "current_dimension_set": "fixed_v1",
    }
    # Check latest observation for dimension set
    for o in reversed(obs):
        ds = o.get("permansor_dimension_set")
        if ds:
            data["readiness"]["current_dimension_set"] = ds
            break

    return data


# ------------------------------------------------------------------
# Cross-client summary
# ------------------------------------------------------------------

def cross_client_summary(clients: list[dict]) -> dict:
    total_obs = sum(c.get("observations", {}).get("total", 0) for c in clients)
    total_scored = sum(c.get("observations", {}).get("scored", 0) for c in clients)

    tiers = {"default": 0, "client": 0, "aggregate": 0}
    for c in clients:
        for module, info in c.get("adaptive", {}).items():
            tier = info.get("tier", "default")
            if tier in tiers:
                tiers[tier] += 1

    # Hook library
    hook_lib = _load_json(P.our_memory_dir() / "hook_library.json")
    hook_count = len(hook_lib) if isinstance(hook_lib, list) else 0

    # Universal patterns
    patterns = _load_json(P.our_memory_dir() / "universal_patterns.json")
    pattern_count = len(patterns) if isinstance(patterns, list) else 0

    # Top/bottom arms across all clients
    all_arms: list[dict] = []
    for c in clients:
        lola = c.get("lola")
        if not lola:
            continue
        for a in lola.get("top_arms", []) + lola.get("bottom_arms", []):
            a["company"] = c["company"]
            all_arms.append(a)

    all_arms_deduped = {(a["company"], a["label"]): a for a in all_arms}
    sorted_arms = sorted(all_arms_deduped.values(), key=lambda a: a.get("avg_reward", 0), reverse=True)

    return {
        "total_observations": total_obs,
        "total_scored": total_scored,
        "client_count": len(clients),
        "adaptive_tiers": tiers,
        "hook_library_size": hook_count,
        "universal_patterns": pattern_count,
        "top_arms": sorted_arms[:5],
        "bottom_arms": sorted_arms[-5:] if len(sorted_arms) >= 5 else [],
    }


# ------------------------------------------------------------------
# Formatters
# ------------------------------------------------------------------

def print_client(c: dict) -> None:
    name = c["company"]
    obs = c["observations"]
    rs = c["reward_stats"]
    eng = c["engagement"]
    cad = c["cadence"]

    print(f"\n  -- {name} --")
    print(f"    Observations:  {obs['total']} total, {obs['scored']} scored ({obs['pct_scored']:.0f}%)")
    print(f"    Reward trend:  {c['reward_sparkline']}  (last 8)")
    print(f"    Reward stats:  mean={rs['mean']:.2f}  median={rs['median']:.2f}  std={rs['std']:.2f}  [{rs['min']:.2f}, {rs['max']:.2f}]")
    print(f"    Engagement:    avg {_fmt_num(eng['avg_impressions'])} impr | {eng['avg_reactions']:.0f} react | {eng['avg_comments']:.1f} comments")
    print(f"    Cadence:       every {cad['avg_days']:.1f} days | {cad['posts_last_7d']} posts last 7d")

    lola = c.get("lola")
    if lola:
        print(f"\n    LOLA ({lola['total_pulls']} pulls, {lola['active']} active, "
              f"{lola['retired']} retired, {lola['icp_boosted']} icp-boosted):")
        for a in lola.get("top_arms", []):
            print(f"      ✦ {a['label']:<35s} {a['pulls']:3d} pulls  avg {a['avg_reward']:+.3f}")
        if lola.get("bottom_arms"):
            for a in lola["bottom_arms"]:
                print(f"      ✧ {a['label']:<35s} {a['pulls']:3d} pulls  avg {a['avg_reward']:+.3f}")
        print(f"      exploration_rate: {lola['exploration_rate']:.0%}")

    adaptive = c.get("adaptive", {})
    if adaptive:
        print(f"\n    Adaptive configs:")
        for module, info in adaptive.items():
            tier = info.get("tier", "default")
            extras = ""
            if module == "permansor" and info.get("pass_threshold"):
                extras = f" threshold={info['pass_threshold']}"
            if module == "constitutional" and info.get("soft_principles"):
                extras = f" soft={info['soft_principles']}"
            print(f"      {module:<16s} {tier}{extras}")

    rd = c.get("readiness", {})
    print(f"\n    Adaptive readiness:")
    perm_need = rd.get("permansor_weights_need", 10)
    const_need = rd.get("constitutional_soft_need", 15)
    emerg_need = rd.get("emergent_dims_need", 40)
    perm_col = rd.get("permansor_dims_collected", 0)
    const_col = rd.get("constitutional_collected", 0)
    perm_status = "READY" if perm_need == 0 else f"{perm_need} more needed"
    const_status = "READY" if const_need == 0 else f"{const_need} more needed"
    emerg_status = "READY" if emerg_need == 0 else f"{emerg_need} more needed"
    print(f"      Permansor weights:  {perm_col}/10 obs ({perm_status})")
    print(f"      Constitutional:     {const_col}/15 obs ({const_status})")
    print(f"      Emergent dims:      {perm_col}/40 obs ({emerg_status})")
    print(f"      Dimension set:      {rd.get('current_dimension_set', 'fixed_v1')}")


def print_summary(summary: dict) -> None:
    print(f"\n{'═' * 62}")
    print(f"  CROSS-CLIENT SUMMARY")
    print(f"    Total observations: {summary['total_observations']} across {summary['client_count']} clients")
    print(f"    Scored: {summary['total_scored']}")
    tiers = summary["adaptive_tiers"]
    print(f"    Adaptive tiers: {tiers.get('client', 0)} client, "
          f"{tiers.get('aggregate', 0)} aggregate, {tiers.get('default', 0)} default")
    print(f"    Hook library: {summary['hook_library_size']} hooks")
    print(f"    Universal patterns: {summary['universal_patterns']}")
    if summary.get("top_arms"):
        print(f"\n    Top arms:")
        for a in summary["top_arms"][:5]:
            print(f"      ✦ {a['label']:<30s} ({a['company']}) avg {a['avg_reward']:+.3f}")
    if summary.get("bottom_arms"):
        print(f"    Bottom arms:")
        for a in summary["bottom_arms"][-5:]:
            print(f"      ✧ {a['label']:<30s} ({a['company']}) avg {a['avg_reward']:+.3f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Amphoreus Learning Dashboard")
    parser.add_argument("--client", help="Show single client")
    parser.add_argument("--all", action="store_true", help="Show all clients")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not args.client and not args.all:
        parser.print_help()
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M %Z")

    # Collect client list
    if args.client:
        companies = [args.client]
    else:
        companies = sorted([
            d.name for d in P.MEMORY_ROOT.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "our_memory"
            and (d / "ruan_mei_state.json").exists()
        ])

    clients = [collect_client(c) for c in companies]

    if args.json:
        output = {
            "generated_at": now,
            "clients": clients,
            "summary": cross_client_summary(clients) if args.all else None,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
        return

    print(f"╔{'═' * 62}╗")
    print(f"║  AMPHOREUS LEARNING DASHBOARD  --  {now:<27s}║")
    print(f"╠{'═' * 62}╣")

    for c in clients:
        print_client(c)

    if args.all:
        print_summary(cross_client_summary(clients))

    print(f"{'═' * 64}")


if __name__ == "__main__":
    main()
