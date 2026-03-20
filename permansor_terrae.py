import os
from google import genai
from google.genai import types

class PermansorTerrae:
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
        directory = f"./client_data/{company_keyword}"
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

        Provide a concise "Fact-Check Report" using the following format:
        - [INTERNAL VERIFICATION]: (Pass/Fail) Brief explanation of whether the post aligns with the client's local transcripts.
        - [EXTERNAL VERIFICATION]: (Pass/Fail) Brief explanation of whether the industry claims/stats hold up to live web search.
        - [CORRECTIONS NEEDED]: List any specific factual errors or hallucinations that need to be fixed. If none, say "None."
        """

        prompt = f"""
        <local_context>
        {local_context}
        </local_context>

        <drafted_post>
        {post_content}
        </drafted_post>
        
        Execute your fact-check.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=system_instruction + "\n\n" + prompt,
                config=search_config
            )
            return response.text
            
        except Exception as e:
            return f"Fact-check failed due to API error: {e}"