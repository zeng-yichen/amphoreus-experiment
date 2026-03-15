import os
import time
import pandas as pd
from google import genai
from google.genai import types

class Mydei:
    """
    Mydei: The Agnostic ABM Target Researcher.
    Ingests unstructured .xlsx files, concatenates row data, and lets Gemini 
    figure out if the target is a person, company, or both before researching.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client()
        else:
            self.client = None
            print("[MYDEI WARNING]: GEMINI_API_KEY environment variable is not set.")

    def research_target(self, target_query):
        """Uses Gemini with Google Search to dynamically identify and profile a target."""
        if not self.client:
            return "Error: Gemini API key missing."

        config = types.GenerateContentConfig(
            temperature=0.3, 
            tools=[{"google_search": {}}]
        )

        prompt = f"""
        Perform deep Account-Based Marketing (ABM) research on the following target: '{target_query}'
        
        This target might be a company, an individual executive, a vague description, or a combination of them. 
        Use your Google Search tool to first identify exactly who or what this target is. Once identified, generate a highly tactical "ABM Target Profile" formatted in Markdown.
        
        Please include the following sections:
        
        ### 1. The Target Overview
        Identify who or what this target is. What is their current market position, core value proposition, or professional background?
        
        ### 2. Recent News & Strategic Shifts
        Detail any recent product launches, funding rounds, leadership changes, or public statements announced in the last 6-12 months regarding this target.
        
        ### 3. Predicted Pain Points
        Based on their industry, competitors, and recent news, what are the 2-3 biggest operational, strategic, or professional challenges this target is likely facing right now?
        
        ### 4. Ingress Strategy
        Provide 2 unique angles a ghostwriter could use to organically catch this target's attention in a LinkedIn post or cold outreach.
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"Mydei API Error for '{target_query}': {e}")
            return f"Error gathering ABM data: {e}"

    def process_abm_files(self, client_keyword, file_paths):
        """Reads any Excel file blindly, combines row text, and generates profiles."""
        if not file_paths:
            print("No files provided to Mydei.")
            return

        out_dir = os.path.join("./client_data", client_keyword, "abm_profiles")
        os.makedirs(out_dir, exist_ok=True)

        for path in file_paths:
            if not path.lower().endswith(('.xlsx', '.xls')):
                print(f"Skipping {os.path.basename(path)}: Not an Excel file.")
                continue
                
            print(f"\n[MYDEI] Reading unstructured target list: {os.path.basename(path)}...")
            
            try:
                # Read completely blindly, assuming no headers at all
                df = pd.read_excel(path, header=None)
                if df.empty:
                    print("File is empty.")
                    continue

                for index, row in df.iterrows():
                    # Drop empty cells (NaN) and combine everything else in the row into a single string
                    row_values = row.dropna().astype(str).tolist()
                    if not row_values:
                        continue
                        
                    target_query = " | ".join(row_values).strip()
                    # Skip rows that are clearly just headers
                    if "company" in target_query.lower() and "name" in target_query.lower():
                        continue

                    print(f"  -> Researching Target: [{target_query}]...")
                    
                    profile_text = self.research_target(target_query)
                    
                    # Create a safe, truncated filename
                    safe_name = "".join([c if c.isalnum() else "_" for c in target_query])
                    safe_name = safe_name[:40].strip("_")
                    filename = f"ABM_{safe_name}.md"
                    
                    filepath = os.path.join(out_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# ABM Target Profile: {target_query}\n\n")
                        f.write(profile_text)
                        
                    print(f"     Saved profile to {filename}")
                    
                    # Sleep briefly to avoid Google Search API rate limits
                    time.sleep(4)

            except Exception as e:
                print(f"Failed to process Excel file {path}: {e}")

        print("\n[MYDEI] ABM Research Complete. Profiles are ready for Phainon.")