import re
import logging
from typing import Dict, List, Optional, Union, Generator
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_BATCH_POSTS = 12


class Cyrene:
    """
    Cyrene: The Master Editor.
    Handles the synthesis of multiple model drafts into a single high-quality 
    baseline, and rewrites it autonomously based on user style instructions or 
    by introducing creative stylistic noise.
    """

    def __init__(self, model_name: str = "claude-opus-4-6"):
        self.model_name = model_name
        self.client = Anthropic()

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 8192) -> str:
        """Calls the Anthropic API using the specified Claude model."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error during LLM call: {e}")
            return f"<error>Failed to generate response: {e}</error>"

    def _format_dict(self, dictionary: Dict[str, str], title: str) -> str:
        formatted = ""
        for key, value in dictionary.items():
            formatted += f"[{title}: {key}]\n{value}\n\n"
        return formatted

    def _parse_xml_tags(self, text: str, tag: str) -> str:
        """Extracts content between specified XML tags, with a fallback for cut-off text."""
        # Try to find the clean open and closed tags first
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
            
        # FALLBACK: If the model got cut off and didn't print the closing tag, 
        # just grab everything after the opening tag to save the recovered text.
        fallback_pattern = f"<{tag}>(.*)"
        fallback_match = re.search(fallback_pattern, text, re.DOTALL)
        
        if fallback_match:
            return fallback_match.group(1).strip() + "\n\n[WARNING: GENERATION CUT OFF DUE TO LENGTH]"
            
        return f"Error: Could not find <{tag}> in response."

    def synthesize_post(self, raw_drafts: Dict[str, str]) -> Dict[str, str]:
        """
        Takes raw drafts from different models and synthesizes them into one masterpiece.
        """
        logger.info("Starting synthesis of raw drafts...")
        
        system_prompt = "You are Cyrene, an elite executive ghostwriter and editor. Your job is to take multiple AI-generated drafts and synthesize them into a single, cohesive, highly-tactical LinkedIn post."
        
        drafts_formatted = self._format_dict(raw_drafts, "DRAFT")

        user_prompt = f"""
        <raw_drafts>
        {drafts_formatted}
        </raw_drafts>

        INSTRUCTIONS:
        Read the provided drafts. They are variations of the same intended LinkedIn post.
        Your goal is to merge the best hooks, the strongest empirical examples, and the tightest conclusions into ONE master draft. 
        Eliminate all generic AI fluff, repetitive transitions, and cliché conclusions.

        Output your response in the following exact XML format:

        <step_1_strategy>
        Briefly explain which elements from which drafts you are keeping and why.
        </step_1_strategy>

        <final_synthesized_post>
        [Your masterfully merged draft goes here]
        </final_synthesized_post>
        """

        raw_response = self._call_llm(system_prompt, user_prompt)
        
        return {
            "strategy": self._parse_xml_tags(raw_response, "step_1_strategy"),
            "synthesized_draft": self._parse_xml_tags(raw_response, "final_synthesized_post")
        }

    def rewrite_single_post(self, post_text: str, style_instruction: str, image_suggestion: str = "", theme: str = "", client_context: str = "") -> dict:
        """
        Method to rewrite a single post using Cyrene's 4-step XML framework.
        
        Args:
            post_text: The original post text to rewrite.
            style_instruction: Stylistic direction for the rewrite.
            image_suggestion: Optional image suggestion to preserve through the rewrite.
            theme: Optional theme/topic of the post to preserve through the rewrite.
            client_context: Optional concatenated transcripts/feedback/accepted posts
                for grounding edits in real client material.
        """
        system_prompt = "You are Cyrene, a meticulous copyeditor. Your job is to completely rewrite a draft stylistically while maintaining 100% of the original factual payload and logical arguments."
        if client_context:
            system_prompt = (
                "You have access to the following client context (interview transcripts, "
                "approved posts, feedback). Use this to ground your edits in what the client "
                "actually said and prefers.\n\n"
                + client_context + "\n\n" + system_prompt
            )
        
        if style_instruction and style_instruction.strip():
            style_directive = f"Apply the following stylistic direction strictly: '{style_instruction.strip()}'"
            analysis_directive = "Analyze the requested style direction and outline how you will apply it."
        else:
            style_directive = "No specific style was provided. You must introduce creative stylistic 'noise'—randomize sentence lengths, swap vocabulary, creatively restructure the flow, and change the emotional undertone slightly to make it feel organic and highly distinct from the original. Do NOT change any facts or the core message. DO NOT change the post length dramatically."
            analysis_directive = "Briefly describe the random stylistic variations and 'noise' you are choosing to apply to this draft."
        
        user_prompt = f"""
        <raw_draft>
        {post_text}
        </raw_draft>

        STYLE DIRECTIVE:
        {style_directive}

        INSTRUCTIONS:
        Output your response in the following exact XML format:

        <step_1_fact_extraction>
        List bullet points of every core argument, statistic, and narrative beat from the raw draft. This is your factual checklist that MUST survive the rewrite.
        </step_1_fact_extraction>

        <step_2_style_approach>
        {analysis_directive}
        </step_2_style_approach>

        <step_3_rewrite_strategy>
        Explain how you will map the facts from Step 1 onto the stylistic approach identified in Step 2.
        </step_3_rewrite_strategy>

        <final_post>
        [Your rewritten, beautifully formatted post goes here]
        </final_post>
        """

        raw_response = self._call_llm(system_prompt, user_prompt)
        
        return {
            "fact_extraction": self._parse_xml_tags(raw_response, "step_1_fact_extraction"),
            "style_analysis": self._parse_xml_tags(raw_response, "step_2_style_approach"),
            "strategy": self._parse_xml_tags(raw_response, "step_3_rewrite_strategy"),
            "final_post": self._parse_xml_tags(raw_response, "final_post"),
            "image_suggestion": image_suggestion.strip() if image_suggestion else "",
            "theme": theme.strip() if theme else "",
        }

    def _parse_batch_post_blocks(self, raw_response: str) -> Dict[int, str]:
        """Extract inner XML for each <post index="n">...</post> block."""
        blocks: Dict[int, str] = {}
        pattern = re.compile(
            r'<post\s+index\s*=\s*["\']?(\d+)["\']?\s*>(.*?)</post>',
            re.DOTALL | re.IGNORECASE,
        )
        for m in pattern.finditer(raw_response):
            idx = int(m.group(1))
            blocks[idx] = m.group(2).strip()
        return blocks

    def rewrite_post_batch(self, posts: List[str], style_instruction: str, image_suggestions: Optional[List[str]] = None, themes: Optional[List[str]] = None, client_context: str = "") -> Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]:
        """
        Rewrite up to MAX_BATCH_POSTS drafts in one model call so cross-post edits
        (deduping, tone alignment, etc.) can use full batch context.
        
        Args:
            posts: List of post texts to rewrite.
            style_instruction: Stylistic direction for the batch rewrite.
            image_suggestions: Optional list of image suggestions, one per post. Preserved through rewrite.
            themes: Optional list of themes/topics, one per post. Preserved through rewrite.
            client_context: Optional concatenated transcripts/feedback/accepted posts
                for grounding edits in real client material.

        Returns:
            batch_coordinator_notes: str — how the batch instruction was applied across posts
            posts: list of dicts with keys index, fact_extraction, style_analysis, strategy, final_post, image_suggestion, theme
        """
        cleaned = [p.strip() for p in posts if p and p.strip()]
        
        img_list = image_suggestions or []
        while len(img_list) < len(cleaned):
            img_list.append("")
        
        theme_list = themes or []
        while len(theme_list) < len(cleaned):
            theme_list.append("")
        if not cleaned:
            raise ValueError("At least one non-empty post is required.")
        if len(cleaned) > MAX_BATCH_POSTS:
            raise ValueError(f"At most {MAX_BATCH_POSTS} posts allowed; got {len(cleaned)}.")

        system_prompt = (
            "You are Cyrene, a meticulous copyeditor. You are given several LinkedIn drafts at once. "
            "Rewrite each one stylistically while preserving factual payload and core arguments. "
            "You MUST read every draft before rewriting any: honor instructions that span posts "
            "(e.g. remove redundancy across posts, align terminology, balance hooks, avoid repeating the same anecdote)."
        )
        if client_context:
            system_prompt = (
                "You have access to the following client context (interview transcripts, "
                "approved posts, feedback). Use this to ground your edits in what the client "
                "actually said and prefers.\n\n"
                + client_context + "\n\n" + system_prompt
            )

        if style_instruction and style_instruction.strip():
            style_directive = f"Apply the following direction across the entire batch: '{style_instruction.strip()}'"
            analysis_directive = "Briefly note how this direction applies to the batch as a whole and any cross-post tradeoffs."
        else:
            style_directive = (
                "No specific style was provided. Introduce creative stylistic variation per post while keeping "
                "the set feeling coherent—vary sentence rhythm and vocabulary, but do NOT change facts or core messages. "
                "Do not make every post identical."
            )
            analysis_directive = "Briefly note stylistic variation choices and any cross-post differentiation."

        numbered = "\n\n".join(
            f'<draft index="{i + 1}">\n{p}\n</draft>' for i, p in enumerate(cleaned)
        )

        user_prompt = f"""
You will rewrite {len(cleaned)} drafts. Indices are 1..{len(cleaned)}.

<all_drafts>
{numbered}
</all_drafts>

STYLE DIRECTIVE:
{style_directive}

OUTPUT RULES:
- First output <batch_rewrite_coordinator>...</batch_rewrite_coordinator> explaining how you applied the directive across ALL drafts (including cross-post edits).
- Then output exactly {len(cleaned)} blocks, one per draft, in order, using this shape for index i = 1..{len(cleaned)}:

<post index="i">
<step_1_fact_extraction>
Concise bullets: core facts and beats for THIS draft only (checklist that must survive).
</step_1_fact_extraction>
<step_2_style_approach>
{analysis_directive} (focused on this draft; keep brief).
</step_2_style_approach>
<step_3_rewrite_strategy>
How this draft’s facts map to the style approach; mention cross-batch choices only when they affect this post.
</step_3_rewrite_strategy>
<final_post>
The rewritten post text only.
</final_post>
</post>

Keep intermediate steps compact so every <final_post> completes. Do not skip any index from 1 to {len(cleaned)}.
"""

        raw_response = self._call_llm(system_prompt, user_prompt, max_tokens=16384)
        coordinator = self._parse_xml_tags(raw_response, "batch_rewrite_coordinator")
        blocks_by_index = self._parse_batch_post_blocks(raw_response)

        results: List[Dict[str, Union[str, int]]] = []
        for i in range(1, len(cleaned) + 1):
            img_sug = img_list[i - 1].strip() if (i - 1) < len(img_list) else ""
            post_theme = theme_list[i - 1].strip() if (i - 1) < len(theme_list) else ""
            inner = blocks_by_index.get(i, "")
            if not inner:
                results.append(
                    {
                        "index": i,
                        "fact_extraction": f"Error: missing <post index=\"{i}\"> in model response.",
                        "style_analysis": "",
                        "strategy": "",
                        "final_post": "",
                        "image_suggestion": img_sug,
                        "theme": post_theme,
                    }
                )
                continue
            results.append(
                {
                    "index": i,
                    "fact_extraction": self._parse_xml_tags(inner, "step_1_fact_extraction"),
                    "style_analysis": self._parse_xml_tags(inner, "step_2_style_approach"),
                    "strategy": self._parse_xml_tags(inner, "step_3_rewrite_strategy"),
                    "final_post": self._parse_xml_tags(inner, "final_post"),
                    "image_suggestion": img_sug,
                    "theme": post_theme,
                }
            )

        return {"batch_coordinator_notes": coordinator, "posts": results}

    def apply_suggestions_batch(
        self,
        posts: List[str],
        per_post_suggestions: Dict[int, List[str]],
        cross_post_suggestions: Optional[List[str]] = None,
        protected_names: Optional[List[str]] = None,
    ) -> Dict[int, str]:
        """Apply targeted revision suggestions without rewriting the posts.

        Unlike rewrite_post_batch, this method is explicitly constrained to
        make ONLY the changes described in the suggestions. Everything else —
        hook, structure, voice, tone, word choice — must be preserved verbatim.

        Parameters
        ----------
        posts : list of post texts (0-indexed)
        per_post_suggestions : dict mapping post index (0-based) → list of
            suggestion strings for that post
        cross_post_suggestions : optional list of cross-post suggestion strings
        protected_names : optional list of ABM target names that must never
            be removed, generalized, or paraphrased

        Returns
        -------
        dict mapping post index (0-based) → revised text. Posts with no
        suggestions are omitted.
        """
        if not per_post_suggestions and not cross_post_suggestions:
            return {}

        numbered = "\n\n".join(
            f'<draft index="{i + 1}">\n{p}\n</draft>' for i, p in enumerate(posts)
        )

        sug_lines = []
        for idx in sorted(per_post_suggestions):
            for s in per_post_suggestions[idx]:
                sug_lines.append(f"- Post {idx + 1}: {s}")
        for s in (cross_post_suggestions or []):
            sug_lines.append(f"- Cross-post: {s}")
        sug_block = "\n".join(sug_lines)

        abm_clause = ""
        if protected_names:
            names_list = "\n".join(f"  - {n}" for n in protected_names)
            abm_clause = (
                " CRITICAL: Some posts contain strategically placed ABM (Account-Based Marketing) "
                "target names. You MUST preserve every mention of these names exactly as written. "
                "Do NOT generalize, paraphrase, or remove them under any circumstances, even if "
                "a revision seems to imply simplifying or cutting that section."
            )

        system_prompt = (
            "You are a surgical copy editor. You receive LinkedIn post drafts and a list of "
            "specific, approved revisions. Your ONLY job is to apply each listed revision. "
            "You must NOT alter anything else. Preserve the exact hook, structure, paragraph "
            "breaks, voice, word choices, and tone of each post except where a revision "
            "explicitly requires a change. If a revision is ambiguous, make the smallest "
            "possible edit that satisfies it." + abm_clause
        )

        abm_rule = ""
        if protected_names:
            names_list = "\n".join(f"  - {n}" for n in protected_names)
            abm_rule = (
                f"- PROTECTED ABM TARGETS — the following company/product names are strategic "
                f"ABM placements. You MUST keep every mention intact. Do NOT generalize them "
                f"(e.g., do not replace 'Mistral's Voxtral TTS' with 'labs are pushing...'). "
                f"Do NOT remove them even to improve flow:\n{names_list}\n"
            )

        user_prompt = f"""
<all_drafts>
{numbered}
</all_drafts>

APPROVED REVISIONS TO APPLY:
{sug_block}

RULES:
{abm_rule}- Apply ONLY the revisions listed above. Touch NOTHING else.
- Do NOT change hooks, openings, or closings unless a revision explicitly targets them.
- Do NOT restructure, rephrase, or "improve" any sentence that is not covered by a revision.
- If a post has no revisions, output it UNCHANGED.
- Output every post that had at least one revision applied, using this format:

<post index="i">
<final_post>
[The post with ONLY the specified revisions applied]
</final_post>
</post>

Omit posts that required no changes. Do not output commentary or reasoning.
"""

        raw_response = self._call_llm(system_prompt, user_prompt, max_tokens=16384)
        blocks = self._parse_batch_post_blocks(raw_response)

        revised: Dict[int, str] = {}
        for batch_idx, inner in blocks.items():
            final = self._parse_xml_tags(inner, "final_post").strip()
            if final and 1 <= batch_idx <= len(posts):
                revised[batch_idx - 1] = final
        return revised

    def rewrite_posts_iteratively(self, full_draft_text: str, style_instruction: str = "") -> Generator[Dict[str, Union[str, int]], None, None]:
        """
        Splits the massive draft block into individual posts, iterating through them 
        one-by-one to prevent context cutoff and improve stylistic quality.
        """
        # Split by the 50-asterisk delimiter from phainon.py
        raw_posts = [p.strip() for p in full_draft_text.split("*" * 50) if p.strip()]
        
        system_prompt = "You are Cyrene, a meticulous copyeditor. Your job is to completely rewrite a draft stylistically while maintaining 100% of the original factual payload and logical arguments."
        
        if style_instruction and style_instruction.strip():
            style_directive = f"Apply the following stylistic direction strictly: '{style_instruction.strip()}'"
            analysis_directive = "Analyze the requested style direction and outline how you will apply it."
        else:
            style_directive = "No specific style was provided. You must introduce creative stylistic 'noise'—randomize sentence lengths, swap vocabulary, creatively restructure the flow, and change the emotional undertone slightly to make it feel organic and highly distinct from the original. Do NOT change any facts or the core message. DO NOT change the post length dramatically."
            analysis_directive = "Briefly describe the random stylistic variations and 'noise' you are choosing to apply to this draft."

        for index, post_text in enumerate(raw_posts):
            user_prompt = f"""
            <raw_draft>
            {post_text}
            </raw_draft>

            STYLE DIRECTIVE:
            {style_directive}

            INSTRUCTIONS:
            Output your response in the following exact XML format:

            <step_1_fact_extraction>
            List bullet points of every core argument, statistic, and narrative beat from the raw draft. This is your factual checklist that MUST survive the rewrite.
            </step_1_fact_extraction>

            <step_2_style_approach>
            {analysis_directive}
            </step_2_style_approach>

            <step_3_rewrite_strategy>
            Explain how you will map the facts from Step 1 onto the stylistic approach identified in Step 2.
            </step_3_rewrite_strategy>

            <final_post>
            [Your rewritten, beautifully formatted post goes here]
            </final_post>
            """

            raw_response = self._call_llm(system_prompt, user_prompt)
            
            yield {
                "index": index + 1,
                "total": len(raw_posts),
                "fact_extraction": self._parse_xml_tags(raw_response, "step_1_fact_extraction"),
                "style_analysis": self._parse_xml_tags(raw_response, "step_2_style_approach"),
                "strategy": self._parse_xml_tags(raw_response, "step_3_rewrite_strategy"),
                "final_post": self._parse_xml_tags(raw_response, "final_post")
            }