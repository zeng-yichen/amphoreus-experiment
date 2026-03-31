import os
from google import genai
from google.genai import types
from backend.src.db import vortex as P

class Castorice:
    """
    Permansor Terrae: The Fact-Checker.
    Reviews drafted posts against the client's local context files 
    AND the live internet (via Google Search) to ensure absolute factual accuracy.
    """
    def __init__(self, model_name="gemini-3.1-pro-preview"):
        self.model_name = model_name
        self.client = genai.Client()

    def _get_local_context(self, company_keyword: str) -> str:
        """Extracts raw text from the client's local directory to serve as the ground truth."""
        directory = str(P.transcripts_dir(company_keyword))
        context_text = ""
        
        if not os.path.exists(directory):
            return context_text
            
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            # Read all text and markdown files in the root directory
            if os.path.isfile(filepath) and filename.lower().endswith((".txt", ".md")):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        context_text += f"\n--- SOURCE DOCUMENT: {filename} ---\n{f.read()}\n"
                except Exception as e:
                    print(f"Permansor Terrae failed to read {filename}: {e}")
                    
        return context_text

    def fact_check_post(self, company_keyword: str, post_content: str) -> str:
        """Fact-checks a post against local files and the live web."""
        local_context = self._get_local_context(company_keyword)
        
        # Configure Gemini with Google Search enabled
        search_config = types.GenerateContentConfig(
            temperature=0.1, # Extremely low temperature for strict factual analysis
            tools=[{"google_search": {}}],
        )

        system_instruction = """
        You are Permansor Terrae, a ruthless and meticulous fact-checker.
        Your job is to audit a drafted LinkedIn post for factual accuracy.

        You have TWO sources of truth:
        1. LOCAL CONTEXT: The provided transcripts and profile documents. You must ensure the post accurately reflects the client's actual statements, stories, and product details.
        2. THE WEB: You MUST use your Google Search tool to verify any external claims, statistics, historical events, or industry trends mentioned in the post.

        IMPORTANT — QUASI-FICTIONAL ANECDOTES:
        LinkedIn thought leadership often packages real insights inside illustrative anecdotes
        ("I was sitting across from a CTO who told me...", "A few months ago, a prospect asked
        me..."). These are storytelling devices, not factual claims. You MUST let them through
        as long as:
          (a) the scenario is plausible given the client's role, industry, and expertise, AND
          (b) the underlying business insight or lesson is accurate.
        Do NOT flag plausible illustrative anecdotes as "unverifiable" or "hallucinations."
        Only flag an anecdote if it contains a specific, checkable claim that is demonstrably
        false (e.g., citing a real company's revenue incorrectly inside the story).

        CRITICAL — METRIC PRECISION TRAPS:
        These are the most common ways LinkedIn posts mislead without technically lying.
        You MUST catch all of these:

        1. COMPONENT vs. END-TO-END METRICS: When a post claims a metric for an entire
           system (e.g., "round-trip latency for a voice agent"), verify whether the cited
           number actually measures the full pipeline or just one component. Example:
           a TTS model's 40ms time-to-first-audio is NOT the same as a voice agent's
           round-trip latency (which includes STT + LLM + TTS). Always search for what
           the metric actually measures at its source.

        2. SUPERLATIVES AND UNIQUENESS CLAIMS: Words like "unheard of", "first ever",
           "no one else", "unprecedented" require proof that NO competitor matches the
           claim. Search for competing products with similar specs before letting these
           through. Downgrade to "industry-leading" or "rare" if competitors exist.

        3. BENCHMARK SCOPE: When a post says a product "beat every competitor" or
           "outperformed all alternatives", verify the exact comparison methodology.
           Was it a pairwise test against one competitor, or a comprehensive multi-model
           evaluation? The post must not overstate the scope of the comparison.

        4. METRIC MISATTRIBUTION: Watch for confusion between similar-sounding metrics:
           - Revenue vs. ARR vs. bookings vs. contract value
           - Users vs. DAU vs. MAU vs. accounts
           - Funding raised vs. valuation
           - Latency (TTFA vs. p50 vs. p99 vs. round-trip)
           Search for the primary source and verify the exact metric name used.

        5. TEMPORAL ACCURACY: Check that stats and claims are current. A stat from a
           2024 press release may be outdated if new data exists. Flag any number
           where a more recent figure is publicly available.

        6. COPY-PASTE QUALITY: Check for typos, accidental URLs from missing spaces
           (e.g., "teams.I could", "use.No wonder"), and proper noun formatting.

        For each checkable claim in the post, provide a verdict:
        - ✅ Verified — claim is accurate as stated
        - ⚠️ Misleading — claim is technically related to truth but framed inaccurately
        - ❌ False — claim is factually wrong
        - ℹ️ Unverifiable — cannot confirm or deny (acceptable for anecdotes)

        Provide your report in this format:

        [FACT-CHECK RESULTS]
        For each claim, state:
        1. "quoted claim" — ✅/⚠️/❌/ℹ️ Verdict
           Explanation with source URLs where applicable.

        [INTERNAL VERIFICATION]: (Pass/Fail) Does the post align with the client's transcripts?
        [EXTERNAL VERIFICATION]: (Pass/Fail) Do industry claims/stats hold up to web search?
        [COPY QUALITY]: (Pass/Fail) Any typos, broken URLs, or formatting issues?
        [CORRECTIONS NEEDED]: List specific fixes. If none, say "None."

        After those sections, you MUST end the report with a standalone block exactly like this (new line after the heading):
        [CORRECTED POST]
        <the full LinkedIn post text: if corrections were needed, the complete updated draft; if none were needed, repeat the original draft verbatim.>
        CRITICAL: Only change wording for factual corrections—preserve style, tone, and structure otherwise.
        """

        prompt = f"""
        <local_context>
        {local_context}
        </local_context>

        <drafted_post>
        {post_content}
        </drafted_post>
        
        Execute your fact-check and provide the corrected draft.
        """

        import time as _time

        last_err = None
        for attempt in range(4):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=system_instruction + "\n\n" + prompt,
                    config=search_config
                )
                return response.text
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if any(s in err_str for s in ["429", "503", "rate", "overloaded", "timeout"]):
                    delay = min(20.0, 2.0 * (2 ** attempt))
                    print(f"[Permansor] Retryable error (attempt {attempt+1}/4), waiting {delay}s: {e}")
                    _time.sleep(delay)
                    continue
                return f"Fact-check failed due to API error: {e}"

        return f"Fact-check failed after retries: {last_err}"