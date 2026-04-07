"""ICP auto-generator — creates free-text ICP definitions from client context.

Reads transcripts, content strategy, references, LinkedIn profile, AND
performs web research on the company/person to ground the ICP in reality.

Without web research, a company named "InnovoCommerce" gets classified as
ecommerce when it's actually clinical biotech. The web search catches this.

Output is written to ``memory/{company}/icp_definition.json``.
"""

import json
import logging
import os

import anthropic

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")


# ------------------------------------------------------------------
# Web research helpers
# ------------------------------------------------------------------

def _web_research(query: str) -> str:
    """Quick web search via Parallel API. Returns excerpt text or empty string."""
    if not PARALLEL_API_KEY:
        return ""
    try:
        import httpx
        resp = httpx.post(
            "https://api.parallel.ai/v1beta/search",
            json={"objective": query, "mode": "fast", "max_results": 5},
            headers={"x-api-key": PARALLEL_API_KEY, "Content-Type": "application/json"},
            timeout=30.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        lines = []
        for r in results[:3]:
            title = r.get("title", "")
            excerpts = r.get("excerpts", [])
            if title:
                lines.append(title)
            for e in excerpts[:1]:
                lines.append(e[:300])
        return "\n".join(lines)
    except Exception as e:
        logger.debug("[icp_generator] Web search failed for '%s': %s", query[:50], e)
        return ""


def _fetch_linkedin_profile_context(company: str) -> str:
    """Load the client's LinkedIn profile summary if available."""
    profile_path = P.memory_dir(company) / "profile.json"
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            parts = []
            if profile.get("headline"):
                parts.append(f"Headline: {profile['headline']}")
            if profile.get("about"):
                parts.append(f"About: {profile['about'][:500]}")
            if profile.get("current_company"):
                parts.append(f"Company: {profile['current_company']}")
            return "\n".join(parts)
        except Exception:
            pass

    # Try linkedin_username for a web lookup
    username_path = P.linkedin_username_path(company)
    if username_path.exists():
        try:
            username = username_path.read_text().strip()
            if username:
                return _web_research(f"LinkedIn {username} profile company industry")
        except Exception:
            pass

    return ""


def _research_company(company: str, transcript_snippet: str = "") -> str:
    """Web-search the company to determine what they actually do.

    Uses company keyword + any company names mentioned in transcripts.
    """
    # Extract company name hints from transcripts
    search_query = f"{company.replace('-', ' ')} company LinkedIn what do they do"

    # If transcripts mention a specific company name, use it
    if transcript_snippet:
        # Look for capitalized multi-word names that might be the company
        words = transcript_snippet[:500].split()
        for i in range(len(words) - 1):
            if words[i][0:1].isupper() and words[i+1][0:1].isupper() and len(words[i]) > 2:
                candidate = f"{words[i]} {words[i+1]}"
                if candidate.lower() not in ("the the", "i was", "we are"):
                    search_query = f"{candidate} company industry products"
                    break

    return _web_research(search_query)


# ------------------------------------------------------------------
# RuanMei post analysis (what does this client actually write about?)
# ------------------------------------------------------------------

def _get_post_topic_summary(company: str) -> str:
    """Summarize what the client's actual posts are about from RuanMei."""
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        scored = [o for o in rm._state.get("observations", []) if o.get("status") == "scored"]
        if len(scored) < 3:
            return ""
        scored.sort(key=lambda o: o.get("reward", {}).get("immediate", 0), reverse=True)
        analyses = [
            o.get("descriptor", {}).get("analysis", "")[:150]
            for o in scored[:5]
            if o.get("descriptor", {}).get("analysis")
        ]
        if analyses:
            return "TOP POST TOPICS (what this client actually writes about):\n" + "\n".join(f"- {a}" for a in analyses)
    except Exception:
        pass
    return ""


# ------------------------------------------------------------------
# Main generator
# ------------------------------------------------------------------

def generate_icp_definition(company: str, force: bool = False) -> dict | None:
    """Generate and persist an ICP definition from client context + web research.

    Gathers context from:
    1. Transcripts (what the client talks about in interviews)
    2. Content strategy (explicit positioning)
    3. References (client-provided articles)
    4. LinkedIn profile (headline, about, company)
    5. Web research on the company (what they actually do)
    6. RuanMei post analysis (what topics their posts cover)

    Returns the generated definition dict, or None if context is insufficient.
    Skips if icp_definition.json already exists (unless force=True).
    """
    icp_path = P.icp_definition_path(company)
    if icp_path.exists() and not force:
        try:
            return json.loads(icp_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    context_parts: list[str] = []
    transcript_snippet = ""

    # 1. Transcripts (recent)
    t_dir = P.transcripts_dir(company)
    if t_dir.exists():
        for f in sorted(t_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            if f.suffix in (".md", ".txt") and f.stat().st_size < 50_000:
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")[:2000]
                    context_parts.append(f"TRANSCRIPT ({f.name}):\n{text}")
                    if not transcript_snippet:
                        transcript_snippet = text
                except Exception:
                    pass

    # 2. Content strategy
    cs_dir = P.content_strategy_dir(company)
    if cs_dir.exists():
        for f in sorted(cs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:1]:
            if f.suffix in (".md", ".txt"):
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")[:3000]
                    context_parts.append(f"CONTENT STRATEGY:\n{text}")
                except Exception:
                    pass

    # 3. References
    refs_dir = P.references_dir(company)
    if refs_dir.exists():
        for f in sorted(refs_dir.iterdir())[:2]:
            if f.suffix in (".md", ".txt"):
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")[:1500]
                    context_parts.append(f"REFERENCE ({f.name}):\n{text}")
                except Exception:
                    pass

    # 4. LinkedIn profile
    profile_ctx = _fetch_linkedin_profile_context(company)
    if profile_ctx:
        context_parts.append(f"LINKEDIN PROFILE:\n{profile_ctx}")

    # 5. Web research on the company
    company_research = _research_company(company, transcript_snippet)
    if company_research:
        context_parts.append(f"WEB RESEARCH (what this company does):\n{company_research}")

    # 6. RuanMei post analysis
    post_topics = _get_post_topic_summary(company)
    if post_topics:
        context_parts.append(post_topics)

    if not context_parts:
        logger.info("[icp_generator] Insufficient context for %s — skipping", company)
        return None

    prompt = (
        "Based on the following client context, generate an Ideal Customer Profile (ICP) "
        "for this person's LinkedIn audience.\n\n"
        "IMPORTANT: The company keyword/slug may not reflect the actual business. "
        "Rely on the transcripts, content strategy, web research, and post topics "
        "to determine what the company actually does and who they sell to. "
        "Do NOT guess from the company name alone.\n\n"
        "Output a JSON object with exactly two keys:\n"
        '- "description": a 2-4 sentence free-text description of the types of people '
        "who SHOULD be engaging with this person's posts (their ideal audience — job titles, "
        "industries, seniority, interests, goals).\n"
        '- "anti_description": a 1-2 sentence description of people who are NOT the target '
        "audience (e.g., job seekers, recruiters, content marketers, bots).\n\n"
        "Be specific to this person's ACTUAL industry and niche (from the evidence, not the name). "
        "Output ONLY valid JSON, nothing else.\n\n"
        + "\n\n---\n\n".join(context_parts)
    )

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        definition = json.loads(raw)
        if "description" not in definition:
            logger.warning("[icp_generator] LLM response missing 'description' for %s", company)
            return None

        icp_path.parent.mkdir(parents=True, exist_ok=True)
        icp_path.write_text(json.dumps(definition, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[icp_generator] Generated ICP definition for %s", company)
        return definition

    except Exception as e:
        logger.warning("[icp_generator] ICP generation failed for %s: %s", company, e)
        return None
