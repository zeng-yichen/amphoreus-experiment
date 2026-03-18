import re
import logging
from typing import Dict, List, Optional, Union, Generator
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Calls the Anthropic API using the specified Claude model."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=8192,
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