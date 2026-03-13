import re
import logging
from typing import Dict, List, Optional, Union
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Cyrene:
    """
    Cyrene: The Master Editor.
    Handles the synthesis of multiple model drafts into a single high-quality 
    baseline, and then rewrites it to structurally mimic reference posts 
    provided by Demiurge.
    """

    def __init__(self, model_name: str = "claude-opus-4-6"):
        self.model_name = model_name
        # Initialize the client (requires ANTHROPIC_API_KEY in your environment variables)
        self.client = Anthropic()

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Calls the Anthropic API using the specified Claude model."""
        logger.info(f"Calling {self.model_name} for text generation...")
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
            formatted += f"--- {title} FROM {key.upper()} ---\n{value}\n\n"
        return formatted

    def _format_list(self, items: List[str], title: str) -> str:
        formatted = ""
        for i, item in enumerate(items):
            formatted += f"--- {title} {i+1} ---\n{item}\n\n"
        return formatted

    def _parse_xml_tags(self, response_text: str, tag: str) -> Optional[str]:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def synthesize_post(self, raw_drafts: Union[Dict[str, str], str]) -> dict:
        """
        Synthesizes multiple drafts (from Phainon/ensemble) to maximize content quality.
        Focuses entirely on logical flow, combining best arguments, and removing redundancies.
        Does NOT apply specific stylistic constraints.
        """
        logger.info("Cyrene is synthesizing drafts for maximum quality...")

        # Normalize input: if a single string is passed, wrap it in a dict
        if isinstance(raw_drafts, str):
            raw_drafts = {"Primary Draft": raw_drafts}

        is_single = len(raw_drafts) == 1
        draft_term = "draft" if is_single else "drafts"

        system_prompt = f"""
        You are Cyrene, the Master Editor. Your job is to synthesize the provided raw {draft_term} into a single, high-quality master draft.
        
        Focus purely on maximizing the factual and argumentative quality of the content. Combine the best insights, ensure strong logical flow, remove redundancies, and create a comprehensive and compelling narrative.
        
        Do NOT attempt to apply any specific stylistic formatting, tone mimicking, or extreme brevity. Focus solely on creating the perfect foundational draft.
        """

        user_prompt = f"""
        <raw_{draft_term}>
        {self._format_dict(raw_drafts, "DRAFT")}
        </raw_{draft_term}>

        INSTRUCTIONS:
        Output your response in the following exact XML format:

        <synthesis_strategy>
        Explain how you are combining the best elements of the provided drafts to maximize overall quality.
        </synthesis_strategy>

        <synthesized_draft>
        [Your high-quality, comprehensive synthesized draft goes here]
        </synthesized_draft>
        """

        raw_response = self._call_llm(system_prompt, user_prompt)
        
        return {
            "strategy": self._parse_xml_tags(raw_response, "synthesis_strategy"),
            "synthesized_draft": self._parse_xml_tags(raw_response, "synthesized_draft") or raw_response
        }

    def rewrite_post(self, draft: str, reference_posts: List[str]) -> dict:
        """
        Takes a highly-quality drafted text and rewrites it to structurally and 
        stylistically mimic the reference posts provided by Demiurge.
        """
        logger.info("Cyrene is performing stylistic rewrite based on reference posts...")

        system_prompt = f"""
        You are Cyrene, the Master Editor. Your job is to rewrite the provided draft into an improved post to suit the user's tastes.
        
        You have been provided with <reference_posts> from our repository. You must meticulously 
        analyze the stylistic DNA of these references: their pacing, paragraph lengths, vocabulary, 
        use of white space, and hook structures.
        
        Your final rewritten post must perfectly mimic the tone, format, and structural rhythm 
        of the <reference_posts>, while ONLY using the factual information provided in the <raw_draft>.

        <rules_of_engagement>
        1. NO CONTENT DRIFT: You may not add new ideas, statistics, anecdotes, or arguments.
        2. NO OMISSIONS: You must include every core argument and data point from the raw draft.
        3. FLAWLESS MIMICRY: The final post should look and feel exactly like the reference posts.
        </rules_of_engagement>
        """

        # Provide a fallback if Demiurge didn't find any references
        if not reference_posts:
            references_formatted = "NO REFERENCES PROVIDED."
        else:
            references_formatted = self._format_list(reference_posts, "REFERENCE POST")

        user_prompt = f"""
        <reference_posts>
        {references_formatted}
        </reference_posts>

        <raw_draft>
        {draft}
        </raw_draft>

        INSTRUCTIONS:
        Output your response in the following exact XML format:

        <step_1_fact_extraction>
        List bullet points of every core argument/statistic from the raw draft. This is your factual checklist.
        </step_1_fact_extraction>

        <step_2_style_analysis>
        Briefly identify the defining stylistic features of the reference posts (sentence length, formatting, emotional tone).
        </step_2_style_analysis>

        <step_3_rewrite_strategy>
        Explain how you will map the facts from Step 1 onto the style identified in Step 2.
        </step_3_rewrite_strategy>

        <final_post>
        [Your rewritten, beautifully formatted post goes here]
        </final_post>
        """

        raw_response = self._call_llm(system_prompt, user_prompt)
        
        return {
            "fact_extraction": self._parse_xml_tags(raw_response, "step_1_fact_extraction"),
            "style_analysis": self._parse_xml_tags(raw_response, "step_2_style_analysis"),
            "strategy": self._parse_xml_tags(raw_response, "step_3_rewrite_strategy"),
            "final_post": self._parse_xml_tags(raw_response, "final_post") or raw_response
        }