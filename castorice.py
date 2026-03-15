import os
from google import genai
from google.genai import types

google_client = genai.Client()

class Castorice:
    """
    Castorice: The Domain Researcher.
    Uses Gemini's native Google Search grounding to build an industry cheat sheet
    for the ghostwriter before they step into an interview.
    """
    def __init__(self, model_name="gemini-3.1-pro-preview"):
        self.model_name = model_name

    def generate_domain_primer(
        self,
        client_name: str,
        company_keyword: str,
        client_context: str = "",
        client_and_icp_summary: str = "",
    ) -> str:
        """
        Generate domain primer. When client_and_icp_summary (Sections 1 and 2 from Aglaea's
        briefing) is provided, use it to define the industry and tailor research—no separate
        industry_hint file needed.
        """
        print(f"Castorice is researching the {company_keyword} industry for {client_name}...")

        search_config = types.GenerateContentConfig(
            temperature=0.4,
            tools=[{"google_search": {}}],
        )

        system_instruction = f"""
        You are Castorice, an elite industry researcher for a ghostwriting agency.
        Your job is to prepare a ghostwriter to interview an executive ({client_name})
        in the '{company_keyword}' space. The ghostwriter has NO background in this industry
        and may be going in blind—they need a clear industry definition, a quick-start cheat sheet,
        and research grounded in the actual market and in what we already know about this client and their ICP.
        """

        has_summary = bool(client_and_icp_summary.strip())
        context_note = "Base Context (from client folder, e.g. transcripts):\n" + (
            client_context.strip() or "(None provided.)"
        )

        if has_summary:
            step1 = f"""
        Step 1 — Use the briefing summary below (Sections 1 and 2: The Client, ICP) to define the industry.
        From this summary, state in one sentence what INDUSTRY or MARKET we are researching. All research
        below must target that industry/market and must be relevant to this client and their ICP.

        --- SECTIONS 1 AND 2 (CLIENT + ICP) ---
        {client_and_icp_summary.strip()}
        --- END ---
        """
        else:
            step1 = f"""
        Step 1 — Define the industry: From the company keyword '{company_keyword}' and any base context below,
        state in one sentence what INDUSTRY or MARKET we are researching. All research below must target this industry/market.
        """

        client_aware_note = (
            "Use the Sections 1+2 summary and Base Context to tailor the Devil's Advocate questions and jargon "
            "to this client's positioning, product, and ICP."
            if (has_summary or client_context.strip()) else
            "When no summary or context is provided, make Devil's Advocate questions generally relevant to the industry."
        )

        prompt = f"""
        {step1}

        {context_note}

        Step 2 — Using your Google Search tool, research the current state of that industry.

        Step 3 — Generate a "Domain Knowledge Primer" formatted in Markdown.
        CRITICAL: Use `###` for the following sub-headers so they nest properly in our main document:

        ### Industry (One Sentence)
        The one-sentence definition of the market/industry you are researching (from Step 1). This orients the ghostwriter immediately.

        ### Blind Interviewer Quick Start
        A short, scannable block so a ghostwriter with zero prep can survive the first 5 minutes. Include:
        - What to understand in the first 5 minutes (one sentence).
        - 2–3 phrases or points they can say to sound credible (not generic).
        - 1–2 things to avoid saying that would mark them as an outsider.
        Keep this to 4–6 bullet points total.

        ### Industry 101
        A simple, analogy-driven explanation of what this industry actually does.

        ### The Jargon Cheat Sheet
        5–7 crucial acronyms or technical terms the ghostwriter MUST know to sound credible.

        ### Current Macro Trends
        What are the 2–3 biggest shifts, controversies, or innovations happening in this space right now (cite recent news)?

        ### Pain Points
        What keeps executives in this industry awake at night?

        ### Devil's Advocate Questions
        3 smart, slightly provocative questions the ghostwriter can ask to get a passionate response from the client. {client_aware_note}
        """

        try:
            response = google_client.models.generate_content(
                model=self.model_name,
                contents=system_instruction + "\n\n" + prompt,
                config=search_config
            )
            return response.text
            
        except Exception as e:
            print(f"Castorice Research Error: {e}")
            return f"Error gathering domain knowledge: {e}"