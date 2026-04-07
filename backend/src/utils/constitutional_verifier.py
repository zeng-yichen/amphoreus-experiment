"""Constitutional Verifier — multi-model ensemble quality gate.

Runs the final draft through a panel of 2-3 models, each evaluating
against the Amphoreus quality constitution (derived from
reading-and-writing-linkedin-content.md). Majority vote on pass/fail
per principle.

Single-model evaluation has blind spots. The diversity of 3 models at
the cheapest tier costs ~$0.003 per post but catches 40-60% more issues
than any single model.

Usage:
    from backend.src.utils.constitutional_verifier import verify_post

    result = verify_post(
        post_text="...",
        company="hensley-biostats",
    )
    # result = {
    #   "passed": True,
    #   "principles": [...],
    #   "violations": [...],
    #   "model_agreement": 0.87,
    # }
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Adaptive config for constitutional principles
# ------------------------------------------------------------------

class ConstitutionalAdaptiveConfig:
    """Learn which principles correlate with engagement.

    Principles where violation has no engagement impact become advisory
    warnings ("soft") instead of hard gates.
    """

    MODULE_NAME = "constitutional"
    # 15 is higher than Cyrene's 10 because binary splits (pass/fail per
    # principle) need more data than continuous correlations. At 10 obs with
    # a principle that passes 80% of the time, the failure group has ~2 data
    # points — Cohen's d is noise. At 15 you get ~3 failures, and the
    # inner min_n=5 gate in correlate_binary_with_engagement adds a second
    # layer of protection. Principles that rarely fail stay hard by default.
    _MIN_OBS = 15

    def get_defaults(self) -> dict:
        return {
            "soft_principles": [],
            "principle_weights": {},
            "max_soft_violations": 0,
            "_tier": "default",
        }

    def sufficient_data(self, company: str) -> bool:
        return len(self._get_obs(company)) >= self._MIN_OBS

    def resolve(self, company: str) -> dict:
        if self.sufficient_data(company):
            try:
                return self._compute(self._get_obs(company), company)
            except Exception as e:
                logger.debug("[ConstitutionalConfig] Client compute failed: %s", e)

        # Try aggregate
        all_obs = self._get_all_obs()
        if len(all_obs) >= self._MIN_OBS:
            try:
                return self._compute(all_obs, "aggregate")
            except Exception:
                pass

        return self.get_defaults()

    def recompute(self, company: str) -> dict:
        return self.resolve(company)

    def _get_obs(self, company: str) -> list[dict]:
        try:
            from backend.src.agents.ruan_mei import RuanMei
            rm = RuanMei(company)
            return [
                o for o in rm._state.get("observations", [])
                if o.get("status") == "scored"
                and o.get("constitutional_results")
                and o.get("reward", {}).get("immediate") is not None
            ]
        except Exception:
            return []

    def _get_all_obs(self) -> list[dict]:
        all_obs = []
        if P.MEMORY_ROOT.exists():
            for d in P.MEMORY_ROOT.iterdir():
                if d.is_dir() and not d.name.startswith(".") and d.name != "our_memory":
                    all_obs.extend(self._get_obs(d.name))
        return all_obs

    def _compute(self, observations: list[dict], label: str) -> dict:
        from backend.src.utils.correlation_analyzer import correlate_binary_with_engagement

        effects = correlate_binary_with_engagement(
            observations,
            attribute_extractor=lambda obs: {
                pid: passed
                for pid, passed in obs.get("constitutional_results", {}).items()
            },
            min_n=5,
        )

        principle_weights = {}
        for pid, stats in effects.items():
            principle_weights[pid] = stats["effect_size"]

        # Soft principles: those where violation has no significant engagement impact
        soft_principles = [pid for pid, stats in effects.items() if not stats["significant"]]

        # Learn pass threshold from engagement data:
        # Compute what constitutional score well-performing posts achieved.
        # Posts above 40th percentile reward → what was their median const score?
        rewards = sorted([o.get("reward", {}).get("immediate", 0) for o in observations])
        learned_threshold = 0.6  # default
        if len(rewards) >= 10:
            cutoff = rewards[int(len(rewards) * 0.4)]
            good_scores = []
            for obs in observations:
                r = obs.get("reward", {}).get("immediate", 0)
                if r >= cutoff:
                    cr = obs.get("constitutional_results", {})
                    if cr:
                        # Simulate constitutional score for this observation
                        score_sum = 0.0
                        w_sum = 0.0
                        for pid, passed in cr.items():
                            w = abs(principle_weights.get(pid, 0.5))
                            score_sum += (1.0 if passed else 0.0) * w
                            w_sum += w
                        if w_sum > 0:
                            good_scores.append(score_sum / w_sum)
            if good_scores:
                good_scores.sort()
                # Threshold = 25th percentile of good posts (be permissive)
                learned_threshold = good_scores[max(0, len(good_scores) // 4)]

        return {
            "soft_principles": soft_principles,
            "principle_weights": principle_weights,
            "max_soft_violations": len(soft_principles),
            "pass_threshold": round(learned_threshold, 3),
            "_tier": "client" if label != "aggregate" else "aggregate",
            "observation_count": len(observations),
        }


def _load_principle_overrides(company: str) -> dict:
    """Load per-client principle overrides from memory/{company}/constitution_overrides.json."""
    path = P.memory_dir(company) / "constitution_overrides.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ------------------------------------------------------------------
# The 8 Constitutional Principles
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Emergent principle discovery
# ------------------------------------------------------------------

_LEARNED_PRINCIPLES_PATH = P.our_memory_dir() / "learned_principles.json"
_PRINCIPLE_DISCOVERY_MIN_OBS = 15
_PRINCIPLE_DISCOVERY_INTERVAL_DAYS = 7
_MIN_LEARNED_PRINCIPLES = 6
_MIN_EVIDENCE_COUNT = 3


def _discover_principles() -> list[dict] | None:
    """Discover quality principles from top vs bottom performers across all clients.

    Returns list of principle dicts, or None if insufficient data.
    """
    # Check if recent discovery exists
    if _LEARNED_PRINCIPLES_PATH.exists():
        try:
            data = json.loads(_LEARNED_PRINCIPLES_PATH.read_text(encoding="utf-8"))
            last = data.get("discovery_date", "")
            if last:
                from datetime import datetime, timezone, timedelta
                dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - dt).days < _PRINCIPLE_DISCOVERY_INTERVAL_DAYS:
                    principles = data.get("principles", [])
                    if len(principles) >= _MIN_LEARNED_PRINCIPLES:
                        usable = [p for p in principles if p.get("evidence_count", 0) >= _MIN_EVIDENCE_COUNT]
                        if len(usable) >= _MIN_LEARNED_PRINCIPLES:
                            return usable
        except Exception:
            pass

    # Collect top/bottom across all clients
    all_top = []
    all_bottom = []
    total_scored = 0

    for d in P.MEMORY_ROOT.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name == "our_memory":
            continue
        state_path = d / "ruan_mei_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            scored = [o for o in state.get("observations", [])
                      if o.get("status") == "scored" and o.get("descriptor", {}).get("analysis")]
            if len(scored) < 5:
                continue
            total_scored += len(scored)
            scored.sort(key=lambda o: o.get("reward", {}).get("immediate", 0))
            n = len(scored)
            bottom = scored[:max(1, n // 4)]
            top = scored[max(0, n - n // 4):]
            for o in top[-4:]:
                all_top.append(o.get("descriptor", {}).get("analysis", "")[:250])
            for o in bottom[:4]:
                all_bottom.append(o.get("descriptor", {}).get("analysis", "")[:250])
        except Exception:
            continue

    if total_scored < _PRINCIPLE_DISCOVERY_MIN_OBS or len(all_top) < 10:
        return None

    top_text = "\n".join(f"[TOP {i+1}] {a}" for i, a in enumerate(all_top[:15]))
    bottom_text = "\n".join(f"[LOW {i+1}] {a}" for i, a in enumerate(all_bottom[:15]))

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": (
                f"You are analyzing LinkedIn post quality across multiple B2B authors.\n\n"
                f"TOP-PERFORMING posts (highest engagement):\n{top_text}\n\n"
                f"LOW-PERFORMING posts (lowest engagement):\n{bottom_text}\n\n"
                f"Identify 8-10 quality PRINCIPLES that distinguish top from bottom performers.\n\n"
                "Rules:\n"
                "- Each principle must be specific enough that an LLM can evaluate a post against it\n"
                "- Each must be about writing quality the author controls, not audience/timing\n"
                "- Include principles the default rubric would miss\n"
                "- A vague principle ('good vibes') is useless — be concrete\n\n"
                "Return JSON array:\n"
                '[{"id": "snake_case_id", "name": "Short Name", '
                '"description": "2-3 sentence description specific enough to evaluate against", '
                '"evidence_count": 4}]\n'
                "evidence_count = how many of the examples above support this principle.\n"
                "Output ONLY the JSON array."
            )}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        principles = json.loads(raw)
        if not isinstance(principles, list) or len(principles) < 5:
            return None

        # Validate: each must have id, name, description
        valid = []
        for p in principles:
            if p.get("id") and p.get("name") and p.get("description") and len(p["description"]) > 20:
                valid.append({
                    "id": p["id"],
                    "name": p["name"],
                    "description": p["description"],
                    "evidence_count": p.get("evidence_count", 1),
                    "source": "discovered",
                })

        if len(valid) < _MIN_LEARNED_PRINCIPLES:
            return None

        # Persist
        from datetime import datetime, timezone
        save_data = {
            "principles": valid,
            "discovery_date": datetime.now(timezone.utc).isoformat(),
            "source_observations": total_scored,
            "source_clients": sum(1 for d in P.MEMORY_ROOT.iterdir()
                                  if d.is_dir() and (d / "ruan_mei_state.json").exists()),
        }
        _LEARNED_PRINCIPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _LEARNED_PRINCIPLES_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(save_data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(_LEARNED_PRINCIPLES_PATH)

        logger.info("[constitutional] Discovered %d principles from %d observations", len(valid), total_scored)
        return valid

    except Exception as e:
        logger.warning("[constitutional] Principle discovery failed: %s", e)
        return None


def _get_active_principles() -> list[dict]:
    """Return the active principle set: learned if available, else fixed 8."""
    learned = None
    if _LEARNED_PRINCIPLES_PATH.exists():
        try:
            data = json.loads(_LEARNED_PRINCIPLES_PATH.read_text(encoding="utf-8"))
            candidates = data.get("principles", [])
            usable = [p for p in candidates if p.get("evidence_count", 0) >= _MIN_EVIDENCE_COUNT]
            if len(usable) >= _MIN_LEARNED_PRINCIPLES:
                learned = usable
        except Exception:
            pass
    return learned if learned else PRINCIPLES


_FIXED_PRINCIPLES = PRINCIPLES = [
    {
        "id": "omi",
        "name": "One Main Idea",
        "description": (
            "The post has one clear intellectual point that every component is "
            "structured around. The hook hints at, introduces, or summarizes the "
            "OMI, and the body fleshes it out. A skimmer can follow the narrative arc."
        ),
    },
    {
        "id": "no_ai_slop",
        "name": "No AI Patterns",
        "description": (
            "No 'It's not X; it's Y' constructions. No casual hyperbole. "
            "No 'Here's the thing' / 'The truth is' / 'Nobody talks about this'. "
            "No excessive em-dashes. No formulaic LinkedIn structure "
            "(hook→vulnerable admission→numbered list→CTA). No 'let that sink in'. "
            "No 'In today's...' / 'In an era of...'."
        ),
    },
    {
        "id": "hook_quality",
        "name": "Hook Piques Curiosity AND Hints at OMI",
        "description": (
            "The first line (under 200 chars) stops the scroll by piquing curiosity "
            "through impressive numbers, subverted assumptions, ICP title callout, "
            "or story climax — AND connects to the post's main idea. Not just "
            "clickbait; the hook and body must be coherent."
        ),
    },
    {
        "id": "body_cohesion",
        "name": "Cohesive Body",
        "description": (
            "Adjacent paragraphs flow into each other. All content clearly ties to "
            "the OMI. A skimmer still gets the point. No tangents, no padding, "
            "no sections that serve a different idea."
        ),
    },
    {
        "id": "no_selling",
        "name": "No Outright Selling",
        "description": (
            "The post does not sell the author's product/service unless the entire "
            "OMI is about their company. No veiled product pitches disguised as "
            "thought leadership."
        ),
    },
    {
        "id": "formatting",
        "name": "Formatting Accentuates Points",
        "description": (
            "Bullets, numbering, or → are used to accentuate key points, not to "
            "fill space. No walls of text. No over-formatted posts with more "
            "structure than substance."
        ),
    },
    {
        "id": "no_empty_cta",
        "name": "No Empty CTA",
        "description": (
            "No 'What do you think? Tell me in the comments.' unless it's a genuine "
            "lead magnet. The post either ends with a strong closing thought or "
            "invites engagement naturally through the content itself."
        ),
    },
    {
        "id": "transcript_sourced",
        "name": "Grounded in Specific Experience",
        "description": (
            "The post reads as a lived experience or specific insight, not generic "
            "industry commentary. The reader can tell this comes from someone who "
            "has actually done the thing, not someone summarizing articles."
        ),
    },
]


# ------------------------------------------------------------------
# Model-specific evaluation
# ------------------------------------------------------------------

# Phase 3: Continuous confidence prompt (replaces binary pass/fail)
_VERIFIER_PROMPT_CONTINUOUS = """\
You are evaluating a LinkedIn post against quality principles. \
For each principle, rate your confidence that the post satisfies it \
on a scale from 0.0 (clear violation) to 1.0 (fully satisfied).

Do NOT use binary pass/fail. Use the full continuous range:
  1.0 = perfectly satisfies this principle
  0.8 = mostly satisfies, minor concerns
  0.5 = borderline, could go either way
  0.2 = mostly violates, with some redeeming aspects
  0.0 = clear violation

PRINCIPLES:
{principles_text}

POST:
{post_text}

Respond with ONLY a JSON object:
{{
  "evaluations": [
    {{"id": "omi", "confidence": 0.85, "note": "Clear OMI about X, though body drifts slightly"}},
    {{"id": "no_ai_slop", "confidence": 0.95, "note": ""}},
    {{"id": "hook_quality", "confidence": 0.3, "note": "Hook is vague — 'Here's what I learned' doesn't pique curiosity"}},
    ...all principles...
  ]
}}
"""

# Legacy binary prompt (fallback for clients without enough data for continuous weights)
_VERIFIER_PROMPT_BINARY = """\
You are evaluating a LinkedIn post against 8 quality principles. \
For each principle, determine if the post PASSES or FAILS.

PRINCIPLES:
{principles_text}

POST:
{post_text}

Respond with ONLY a JSON object:
{{
  "evaluations": [
    {{"id": "omi", "passed": true, "note": "Clear OMI about X"}},
    {{"id": "no_ai_slop", "passed": true, "note": ""}},
    {{"id": "hook_quality", "passed": false, "note": "Hook is vague"}},
    ...all 8 principles...
  ]
}}

Be strict. If in doubt, FAIL. Better to flag a borderline issue than let it through.
"""

_MIN_OBS_FOR_CONTINUOUS_CONST = 10  # use continuous confidence above this


def _format_principles(principle_set: list[dict] | None = None) -> str:
    ps = principle_set or _get_active_principles()
    lines = []
    for p in ps:
        lines.append(f"- **{p['id']}** ({p['name']}): {p['description']}")
    return "\n".join(lines)


def _evaluate_with_claude(post_text: str, use_continuous: bool = False, principle_set: list[dict] | None = None) -> dict | None:
    """Evaluate with Claude."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        template = _VERIFIER_PROMPT_CONTINUOUS if use_continuous else _VERIFIER_PROMPT_BINARY
        prompt = template.format(principles_text=_format_principles(principle_set), post_text=post_text)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_evaluation(resp.content[0].text.strip(), "claude", use_continuous)
    except Exception as e:
        logger.warning("[constitutional_verifier] Claude evaluation failed: %s", e)
        return None


def _evaluate_with_gemini(post_text: str, use_continuous: bool = False, principle_set: list[dict] | None = None) -> dict | None:
    """Evaluate with Gemini Flash."""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client()
        template = _VERIFIER_PROMPT_CONTINUOUS if use_continuous else _VERIFIER_PROMPT_BINARY
        prompt = template.format(principles_text=_format_principles(principle_set), post_text=post_text)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return _parse_evaluation(resp.text.strip(), "gemini", use_continuous)
    except Exception as e:
        logger.warning("[constitutional_verifier] Gemini evaluation failed: %s", e)
        return None


def _evaluate_with_gpt(post_text: str, use_continuous: bool = False, principle_set: list[dict] | None = None) -> dict | None:
    """Evaluate with GPT-4o-mini."""
    try:
        from openai import OpenAI
        client = OpenAI()
        template = _VERIFIER_PROMPT_CONTINUOUS if use_continuous else _VERIFIER_PROMPT_BINARY
        prompt = template.format(principles_text=_format_principles(principle_set), post_text=post_text)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_evaluation(resp.choices[0].message.content.strip(), "gpt", use_continuous)
    except Exception as e:
        logger.warning("[constitutional_verifier] GPT evaluation failed: %s", e)
        return None


def _parse_evaluation(raw: str, model_name: str, use_continuous: bool = False) -> dict | None:
    """Parse model response into structured evaluations.

    Handles both continuous (confidence float) and binary (passed bool) formats.
    """
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        evals = data.get("evaluations", [])

        result = {"model": model_name, "evaluations": {}}
        for ev in evals:
            pid = ev.get("id", "")
            if not pid:
                continue

            if use_continuous:
                # Continuous: confidence is a float 0.0-1.0
                conf = ev.get("confidence")
                if conf is None:
                    # Fallback: convert binary to continuous
                    conf = 0.9 if ev.get("passed", True) else 0.1
                else:
                    conf = max(0.0, min(1.0, float(conf)))
                result["evaluations"][pid] = {
                    "confidence": conf,
                    "passed": conf >= 0.5,  # backward compat
                    "note": ev.get("note", ""),
                }
            else:
                # Binary: passed is bool
                result["evaluations"][pid] = {
                    "passed": bool(ev.get("passed", True)),
                    "confidence": 1.0 if ev.get("passed", True) else 0.0,
                    "note": ev.get("note", ""),
                }

        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("[constitutional_verifier] Parse failed for %s: %s", model_name, e)
        return None


# ------------------------------------------------------------------
# Ensemble voting
# ------------------------------------------------------------------

def verify_post(
    post_text: str,
    company: str = "",
    models: list[str] | None = None,
) -> dict:
    """Run multi-model constitutional verification.

    Args:
        post_text: The final post text to verify.
        company: Client company keyword (for logging).
        models: Which models to use. Default: ["claude", "gemini", "gpt"].
                Pass fewer for speed/cost tradeoff.

    Returns:
        {
            "passed": bool,           # True if no principle fails by majority vote
            "principles": [...],      # Per-principle results with votes
            "violations": [...],      # Principles that failed
            "model_agreement": float, # 0-1 agreement score across models
            "models_used": int,
        }
    """
    if not post_text or not post_text.strip():
        return _empty_result("No post text provided.")

    if models is None:
        models = ["claude", "gemini", "gpt"]

    # Load adaptive config + manual overrides
    adaptive = ConstitutionalAdaptiveConfig().resolve(company) if company else {}
    principle_weights = adaptive.get("principle_weights", {})
    overrides = _load_principle_overrides(company) if company else {}
    disabled_principles = set(overrides.get("disabled_principles", []))

    # Determine if we use continuous confidence or binary pass/fail
    obs_count = adaptive.get("observation_count", 0)
    use_continuous = obs_count >= _MIN_OBS_FOR_CONTINUOUS_CONST

    # Get active principles (learned or fixed 8)
    principle_set = _get_active_principles()
    active_principles = [p for p in principle_set if p["id"] not in disabled_principles]

    evaluators = {
        "claude": lambda txt: _evaluate_with_claude(txt, use_continuous, active_principles),
        "gemini": lambda txt: _evaluate_with_gemini(txt, use_continuous, active_principles),
        "gpt": lambda txt: _evaluate_with_gpt(txt, use_continuous, active_principles),
    }

    # Run evaluations in parallel
    model_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for m in models:
            if m in evaluators:
                futures[executor.submit(evaluators[m], post_text)] = m

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                result = future.result()
                if result:
                    model_results.append(result)
            except Exception as e:
                logger.warning("[constitutional_verifier] %s failed: %s", model_name, e)

    if not model_results:
        return _empty_result("All model evaluations failed.")

    # Score each principle — continuous or binary depending on mode
    principle_results = []
    total_agreements = 0
    total_votes = 0
    weighted_score_sum = 0.0
    weight_sum = 0.0

    for p in active_principles:
        pid = p["id"]
        confidences: list[float] = []
        notes: list[str] = []

        for mr in model_results:
            ev = mr.get("evaluations", {}).get(pid)
            if ev:
                conf = ev.get("confidence", 1.0 if ev.get("passed", True) else 0.0)
                confidences.append(conf)
                if conf < 0.5 and ev.get("note"):
                    notes.append(f"[{mr['model']}] {ev['note']}")

        if not confidences:
            continue

        # Mean confidence across models
        mean_conf = sum(confidences) / len(confidences)
        # Agreement: how similar are the models' confidences
        if len(confidences) >= 2:
            conf_range = max(confidences) - min(confidences)
            agreement_score = 1.0 - conf_range  # 1.0 = perfect agreement
        else:
            agreement_score = 1.0

        total_agreements += agreement_score
        total_votes += 1

        # Continuous weight for this principle (from engagement correlation)
        # weight=1.0 means violation strongly predicts poor engagement
        # weight=0.0 means violation has no engagement impact
        weight = abs(principle_weights.get(pid, 0.5))  # default 0.5 = moderate

        weighted_score_sum += mean_conf * weight
        weight_sum += weight

        principle_results.append({
            "id": pid,
            "name": p["name"],
            "confidence": round(mean_conf, 3),
            "weight": round(weight, 3),
            "passed": mean_conf >= 0.5,  # backward compat
            "votes_pass": sum(1 for c in confidences if c >= 0.5),
            "votes_fail": sum(1 for c in confidences if c < 0.5),
            "notes": notes,
            "soft": weight < 0.2,  # derived from continuous weight, not categorical
        })

    # Constitutional score: weighted average of principle confidences
    constitutional_score = weighted_score_sum / max(weight_sum, 1e-8)

    # Threshold: learned from engagement data when available, else 0.6
    const_threshold = adaptive.get("pass_threshold", 0.6)
    overall_passed = constitutional_score >= const_threshold

    # Violations: sorted by impact (weight × severity), highest first.
    # No hard/soft categorical split — weight is continuous.
    all_low = [p for p in principle_results if p["confidence"] < 0.5]
    all_low.sort(key=lambda p: p["weight"] * (1.0 - p["confidence"]), reverse=True)

    # Backward-compat: derive hard/soft from weight threshold for display only
    hard_violations = [p for p in all_low if p["weight"] >= 0.2]
    soft_violations = [p for p in all_low if p["weight"] < 0.2]

    agreement = total_agreements / max(total_votes, 1)

    logger.info(
        "[constitutional_verifier] %s: %s (score=%.2f, threshold=%.2f, %d low-conf principles, "
        "%.0f%% agreement, %d models, %s)",
        company, "PASS" if overall_passed else "FAIL",
        constitutional_score, const_threshold,
        len(all_low), agreement * 100, len(model_results),
        "continuous" if use_continuous else "binary",
    )

    return {
        "passed": overall_passed,
        "constitutional_score": round(constitutional_score, 3),
        "threshold": const_threshold,
        "principles": principle_results,
        "violations": hard_violations + soft_violations,
        "hard_violations": hard_violations,
        "soft_violations": soft_violations,
        "disabled_principles": list(disabled_principles),
        "model_agreement": round(agreement, 3),
        "models_used": len(model_results),
        "mode": "continuous" if use_continuous else "binary",
    }


def _empty_result(reason: str) -> dict:
    return {
        "passed": True,  # Don't block on failure
        "principles": [],
        "violations": [],
        "model_agreement": 0.0,
        "models_used": 0,
        "error": reason,
    }
