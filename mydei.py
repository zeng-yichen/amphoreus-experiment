import os
from google import genai
from google.genai import types

class Mydei:
    """
    Mydei: The Autonomous ABM Target Researcher.
    Scans client documents, identifies potential ABM targets, and uses Gemini 
    with Google Search to generate an ABM briefing for post generation.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client()
        else:
            self.client = None
            print("[MYDEI WARNING]: GEMINI_API_KEY environment variable is not set.")

    def generate_abm_briefing(self, company_keyword):
        if not self.client:
            return "[MYDEI WARNING] Gemini API key missing."

        directory = f"./client_data/{company_keyword}"
        if not os.path.exists(directory):
            return "[MYDEI WARNING] Client directory missing."

        # 1. Gather context from client documents
        context_text = ""
        for root, dirs, files in os.walk(directory):
            # Skip output and system directories to prevent circular loops
            if any(skip in root.split(os.sep) for skip in ['output', 'as_written', 'abm_profiles']): 
                continue 
            
            for file in files:
                if file.lower().endswith(('.txt', '.md', '.csv', '.json')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            context_text += f"\n--- DOCUMENT: {file} ---\n{f.read()}\n"
                    except Exception:
                        pass

        if not context_text.strip():
            return "[MYDEI WARNING] No readable documents."

        # 2. Configure Gemini with Live Search
        config = types.GenerateContentConfig(
            temperature=0.3, 
            tools=[{"google_search": {}}]
        )

        prompt = f"""
        You are Mydei, an expert Account-Based Marketing (ABM) researcher.
        Read the following client documents. Identify any specific companies, brands, or executives explicitly mentioned or heavily implied as ideal prospects, clients, or ABM targets.

        CLIENT DOCUMENTS:
        {context_text}

        If you CANNOT find any specific named companies or individuals to target in these documents, reply EXACTLY with: "NO ABM TARGETS FOUND."

        If you DO find targets, pick up to 2 of the most prominent ones. For each, use your Google Search tool to research them on the live internet and provide a highly tactical "ABM Target Profile" formatted in Markdown. 
        
        For each target, include:
        ### [Target Name]
        1. Target Overview
        2. Recent News & Strategic Shifts (last 6-12 months)
        3. Predicted Pain Points
        4. Ingress Strategy (2 unique angles to catch their attention in a LinkedIn post).
        """

        try:
            print(f"[MYDEI] Scanning documents for '{company_keyword}' to extract and research ABM targets...")
            response = self.client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"Mydei API Error: {e}")
            return "[MYDEI WARNING] API Error."