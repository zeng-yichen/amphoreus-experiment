import os
import time
import docx
import concurrent.futures
from google import genai
from google.genai import types
from anthropic import Anthropic
from openai import OpenAI

from castorice import Castorice

google_client = genai.Client()
anthropic_client = Anthropic()
openai_client = OpenAI()

def get_local_context(directory, skip_files):
    """Extracts raw text from local files for non-Google models."""
    context_text = ""
    if not os.path.exists(directory):
        return context_text
    for filename in os.listdir(directory):
        if filename in skip_files: continue
        filepath = os.path.join(directory, filename)
        
        if filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                context_text += f"\n--- DOCUMENT: {filename} ---\n{f.read()}\n"
                
    return context_text

def upload_and_wait(directory, client, skip_files):
    """Helper function to convert docs, upload files, and wait for processing."""
    uploaded_files = []
    if not os.path.exists(directory):
        return uploaded_files
        
    for filename in os.listdir(directory):
        if filename.lower().endswith(".docx"):
            filepath = os.path.join(directory, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_filepath = os.path.join(directory, txt_filename)
            
            if not os.path.exists(txt_filepath):
                print(f"Converting {filename} to TXT...")
                doc = docx.Document(filepath)
                with open(txt_filepath, "w", encoding="utf-8") as f:
                    f.write("\n".join([p.text for p in doc.paragraphs]))
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.lower().endswith(".docx"):
            continue
            
        if os.path.isfile(filepath) and filename not in skip_files:
            print(f"Uploading {filename} from {os.path.basename(directory)}...")
            f = client.files.upload(file=filepath)
            uploaded_files.append(f)
            
    for i, f in enumerate(uploaded_files):
        while f.state.name == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        uploaded_files[i] = f
        
    return uploaded_files

def run_gemini_briefing_workflow(client_name, company_keyword, output_filepath, base_files, blocked_files):
    print("Files ready. Initializing Gemini Chat Session...")
    all_uploaded_files = base_files + blocked_files
    
    chat = google_client.chats.create(
        model="gemini-3.1-pro-preview",
        config=types.GenerateContentConfig(temperature=0.7)
    )

    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"RUAN MEI BRIEFING (GEMINI): {client_name.upper()}\n")
        out_file.write("="*50 + "\n\n")

        print("\nGemini Step 1: Feeding base context...")
        prompt_1 = f"""
        I am attaching several files. Among these are interview transcripts between our organization Virio 
        and a startup founder/executive named {client_name}, as well as a PDF export of their LinkedIn profile. 
        Comprehend the LinkedIn profile and these interview transcripts thoroughly before we proceed.
        """
        response_1 = chat.send_message([prompt_1] + base_files)
        out_file.write(f"--- STEP 1: CONTEXT INGESTION ---\n{response_1.text}\n\n")

        if blocked_files:
            print("Gemini Step 1.1: Analyzing Rejected Posts...")
            prompt_1a = """
            I am attaching files containing LinkedIn posts this client has REJECTED.
            What topics, angles, or tones does the client dislike? We must avoid pulling the interview in these directions.
            """
            response_1a = chat.send_message([prompt_1a] + blocked_files)
            out_file.write(f"--- STEP 1.1: AVOIDANCE TERRITORY (REJECTED POSTS) ---\n{response_1a.text}\n\n")

        print("Gemini Step 2: Analyzing ICP...")
        prompt_2 = f"First, what is {client_name}'s ideal customer profile? What is his/her product? How would his/her product and philosophies resonate with his/her ideal customer?"
        response_2 = chat.send_message(prompt_2)
        out_file.write(f"--- STEP 2: ICP & PRODUCT ANALYSIS ---\n{response_2.text}\n\n")

        print("Gemini Step 3: Triggering Castorice for Domain Primer...")
        castorice = Castorice(model_name="gemini-3.1-pro-preview")
        domain_primer = castorice.generate_domain_primer(
            client_name=client_name,
            company_keyword=company_keyword,
            client_context=response_1.text,
            client_and_icp_summary=response_2.text
        )
        out_file.write(f"--- STEP 3: DOMAIN KNOWLEDGE PRIMER (CASTORICE) ---\n{domain_primer}\n\n")

        print("Gemini Step 4: Generating Dynamic Interview Script...")
        prompt_4 = f"""
        Now, equip the ghostwriter for their next interview with {client_name}. 
        Instead of a static list of questions, create a 'Dynamic Interview Pseudo-Script'. This should act as a conversational decision tree that guides the ghostwriter organically through the interview.

        CRITICAL INSTRUCTION: Ensure the topics and angles explored here are NET-NEW. Do NOT redundantly ask questions about stories, topics, or themes that have already been exhausted in the past transcripts.

        Format the script with the following sections:
        
        ### Phase 1: The Opener
        Provide ONE high-impact, open-ended opening question related to {company_keyword} or their recent shifts that forces the client off autopilot.
        
        ### Phase 2: The Conversation Tree
        Provide 6 distinct branching paths based on how the client answers the opener. Each path should have 6 follow-up questions.
        ALL QUESTIONS SHOULD BE WRITTEN IN STRAIGHTFORWARD, COLLOQUIAL, NATURAL CONVERSATION LANGUAGE. Questions should extract storytelling x virality, hot takes x thought leadership opinions, and overall content that will increase the brand recognition and image of the client and their company.
        FOR EACH QUESTION, PROVIDE THE RELEVANCE OF THE QUESTION TO THE CLIENT'S ICP.

        ### Phase 3: Digging for Stories (The "Yes, and..." technique)
        Provide 10 tactical follow-up prompts the ghostwriter can use at any time to transition from high-level philosophy into concrete, ghostwriting-ready anecdotes.
        """
        response_4 = chat.send_message(prompt_4)
        out_file.write(f"--- STEP 4: DYNAMIC INTERVIEW SCRIPT ---\n{response_4.text}\n\n")

        print("Gemini Step 5: Executive Polish...")
        prompt_5 = "Review everything we've discussed. Synthesize it into a clean, executive summary."
        response_5 = chat.send_message(prompt_5)
        out_file.write(f"--- STEP 5: EXECUTIVE SUMMARY ---\n{response_5.text}\n\n")

    print("\nCleaning up files from Google servers...")
    for f in all_uploaded_files:
        try:
            google_client.files.delete(name=f.name)
        except Exception as e:
            pass

def run_gpt5_briefing_workflow(client_name, company_keyword, output_filepath, base_text, blk_text):
    print("Initializing GPT-5 Chat Session...")
    messages = [{"role": "system", "content": "You are a professional LinkedIn ghostwriter and interviewer."}]
    
    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"RUAN MEI BRIEFING (GPT-5): {client_name.upper()}\n")
        out_file.write("="*50 + "\n\n")

        print("\nGPT-5 Step 1: Feeding context...")
        prompt_1 = f"Comprehend the LinkedIn profile and interview transcripts for {client_name}."
        messages.append({"role": "user", "content": prompt_1 + f"\n\nFILES:\n{base_text}"})
        resp_1 = openai_client.chat.completions.create(model="gpt-5", messages=messages)
        messages.append({"role": "assistant", "content": resp_1.choices[0].message.content})
        out_file.write(f"--- STEP 1: CONTEXT INGESTION ---\n{resp_1.choices[0].message.content}\n\n")

        if blk_text:
            print("GPT-5 Step 1.1: Analyzing Rejected Posts...")
            messages.append({"role": "user", "content": "Review these REJECTED posts. What topics or angles does the client dislike? We must avoid these in future interviews." + f"\n\nREJECTED POSTS:\n{blk_text}"})
            resp_1a = openai_client.chat.completions.create(model="gpt-5", messages=messages)
            messages.append({"role": "assistant", "content": resp_1a.choices[0].message.content})
            out_file.write(f"--- STEP 1.1: AVOIDANCE TERRITORY (REJECTED POSTS) ---\n{resp_1a.choices[0].message.content}\n\n")

        print("GPT-5 Step 2: Analyzing ICP...")
        messages.append({"role": "user", "content": f"What is {client_name}'s ideal customer profile and product?"})
        resp_2 = openai_client.chat.completions.create(model="gpt-5", messages=messages)
        messages.append({"role": "assistant", "content": resp_2.choices[0].message.content})
        out_file.write(f"--- STEP 2: ICP & PRODUCT ANALYSIS ---\n{resp_2.choices[0].message.content}\n\n")

        print("GPT-5 Step 3: Triggering Castorice for Domain Primer...")
        castorice = Castorice()
        domain_primer = castorice.generate_domain_primer(client_name, company_keyword, resp_1.choices[0].message.content, resp_2.choices[0].message.content)
        messages.append({"role": "assistant", "content": f"[SYSTEM INJECTED DOMAIN KNOWLEDGE]:\n{domain_primer}"})
        out_file.write(f"--- STEP 3: DOMAIN KNOWLEDGE PRIMER (CASTORICE) ---\n{domain_primer}\n\n")

        print("GPT-5 Step 4: Generating Dynamic Interview Script...")
        prompt_4 = f"""
        Now, equip the ghostwriter for their next interview with {client_name}. 
        Instead of a static list of questions, create a 'Dynamic Interview Pseudo-Script'. This should act as a conversational decision tree that guides the ghostwriter organically through the interview.

        CRITICAL INSTRUCTION: Ensure the topics and angles explored here are NET-NEW. Do NOT redundantly ask questions about stories, topics, or themes that have already been exhausted in the past transcripts.

        Format the script with the following sections:
        
        ### Phase 1: The Opener
        Provide ONE high-impact, open-ended opening question related to {company_keyword} or their recent shifts that forces the client off autopilot.
        
        ### Phase 2: The Conversation Tree
        Provide 3 distinct branching paths based on how the client answers the opener. 
        - IF the client focuses on [Topic A], THEN the ghostwriter should pivot and ask: [Specific Follow-up Question].
        - IF the client focuses on [Topic B], THEN the ghostwriter should pivot and ask: [Specific Follow-up Question].
        - IF the client gives a short/generic answer, THEN use this fallback probe: [Specific Follow-up Question].

        ### Phase 3: Digging for Stories (The "Yes, and..." technique)
        Provide 3 tactical follow-up prompts the ghostwriter can use at any time to transition from high-level philosophy into concrete, ghostwriting-ready anecdotes.

        ### Phase 4: The Anchor
        Provide one closing question designed to reliably tie the conversation back to their core product and Ideal Customer Profile (ICP).
        """
        messages.append({"role": "user", "content": prompt_4})
        resp_4 = openai_client.chat.completions.create(model="gpt-5", messages=messages)
        messages.append({"role": "assistant", "content": resp_4.choices[0].message.content})
        out_file.write(f"--- STEP 4: DYNAMIC INTERVIEW SCRIPT ---\n{resp_4.choices[0].message.content}\n\n")

        print("GPT-5 Step 5: Executive Polish...")
        messages.append({"role": "user", "content": "Synthesize everything into a clean, executive summary."})
        resp_5 = openai_client.chat.completions.create(model="gpt-5", messages=messages)
        out_file.write(f"--- STEP 5: EXECUTIVE SUMMARY ---\n{resp_5.choices[0].message.content}\n\n")

def run_claude_briefing_workflow(client_name, company_keyword, output_filepath, base_text, blk_text):
    print("Initializing Claude Chat Session...")
    messages = []
    sys_prompt = "You are a professional LinkedIn ghostwriter and interviewer."
    
    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"RUAN MEI BRIEFING (CLAUDE): {client_name.upper()}\n")
        out_file.write("="*50 + "\n\n")

        print("\nClaude Step 1: Feeding context...")
        prompt_1 = f"Comprehend the LinkedIn profile and interview transcripts for {client_name} thoroughly."
        messages.append({"role": "user", "content": prompt_1 + f"\n\nFILES:\n{base_text}"})
        resp_1 = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
        messages.append({"role": "assistant", "content": resp_1.content[0].text})
        out_file.write(f"--- STEP 1: CONTEXT INGESTION ---\n{resp_1.content[0].text}\n\n")

        if blk_text:
            print("Claude Step 1.1: Analyzing Rejected Posts...")
            messages.append({"role": "user", "content": "Review these REJECTED posts. What topics or angles does the client dislike? We must avoid these in future interviews." + f"\n\nREJECTED POSTS:\n{blk_text}"})
            resp_1a = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
            messages.append({"role": "assistant", "content": resp_1a.content[0].text})
            out_file.write(f"--- STEP 1.1: AVOIDANCE TERRITORY (REJECTED POSTS) ---\n{resp_1a.content[0].text}\n\n")

        print("Claude Step 2: Analyzing ICP...")
        messages.append({"role": "user", "content": f"What is {client_name}'s ideal customer profile and product?"})
        resp_2 = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
        messages.append({"role": "assistant", "content": resp_2.content[0].text})
        out_file.write(f"--- STEP 2: ICP & PRODUCT ANALYSIS ---\n{resp_2.content[0].text}\n\n")

        print("Claude Step 3: Triggering Castorice for Domain Primer...")
        castorice = Castorice()
        domain_primer = castorice.generate_domain_primer(client_name, company_keyword, resp_1.content[0].text, resp_2.content[0].text)
        messages.append({"role": "assistant", "content": f"[SYSTEM INJECTED DOMAIN KNOWLEDGE]:\n{domain_primer}"})
        out_file.write(f"--- STEP 3: DOMAIN KNOWLEDGE PRIMER (CASTORICE) ---\n{domain_primer}\n\n")

        print("Claude Step 4: Generating Dynamic Interview Script...")
        prompt_4 = f"""
        Now, equip the ghostwriter for their next interview with {client_name}. 
        Instead of a static list of questions, create a 'Dynamic Interview Pseudo-Script'. This should act as a conversational decision tree that guides the ghostwriter organically through the interview.

        CRITICAL INSTRUCTION: Ensure the topics and angles explored here are NET-NEW. Do NOT redundantly ask questions about stories, topics, or themes that have already been exhausted in the past transcripts.

        Format the script with the following sections:
        
        ### Phase 1: The Opener
        Provide ONE high-impact, open-ended opening question related to {company_keyword} or their recent shifts that forces the client off autopilot.
        
        ### Phase 2: The Conversation Tree
        Provide 3 distinct branching paths based on how the client answers the opener. 
        - IF the client focuses on [Topic A], THEN the ghostwriter should pivot and ask: [Specific Follow-up Question].
        - IF the client focuses on [Topic B], THEN the ghostwriter should pivot and ask: [Specific Follow-up Question].
        - IF the client gives a short/generic answer, THEN use this fallback probe: [Specific Follow-up Question].

        ### Phase 3: Digging for Stories (The "Yes, and..." technique)
        Provide 3 tactical follow-up prompts the ghostwriter can use at any time to transition from high-level philosophy into concrete, ghostwriting-ready anecdotes.

        ### Phase 4: The Anchor
        Provide one closing question designed to reliably tie the conversation back to their core product and Ideal Customer Profile (ICP).
        """
        messages.append({"role": "user", "content": prompt_4})
        resp_4 = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
        messages.append({"role": "assistant", "content": resp_4.content[0].text})
        out_file.write(f"--- STEP 4: DYNAMIC INTERVIEW SCRIPT ---\n{resp_4.content[0].text}\n\n")

        print("Claude Step 5: Executive Polish...")
        messages.append({"role": "user", "content": "Synthesize everything into a clean, executive summary."})
        resp_5 = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
        out_file.write(f"--- STEP 5: EXECUTIVE SUMMARY ---\n{resp_5.content[0].text}\n\n")

def generate_briefing(client_name, company_keyword, model_choice="All (Ensemble)"):
    directory_path = f"./client_data/{company_keyword}"
    output_path = os.path.join(directory_path, "output")
    blocked_path = os.path.join(directory_path, "rejected")
    
    os.makedirs(output_path, exist_ok=True)
    
    google_output_filepath = os.path.join(output_path, f"{company_keyword}_gemini_briefing.md")
    gpt_output_filepath = os.path.join(output_path, f"{company_keyword}_gpt_briefing.md")
    claude_output_filepath = os.path.join(output_path, f"{company_keyword}_claude_briefing.md")
    final_output_filepath = os.path.join(output_path, f"{company_keyword}_briefing.md")

    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    # 1. Load context
    base_files, blocked_files = [], []
    local_context, blk_posts = "", ""
    
    if model_choice in ["All (Ensemble)", "Gemini 3.1 Pro"]:
        base_files = upload_and_wait(directory_path, google_client, skip_files=[f for f in os.listdir(directory_path) if f.startswith(company_keyword)])
        blocked_files = upload_and_wait(blocked_path, google_client, skip_files=[])
        
    if model_choice in ["All (Ensemble)", "GPT-5", "Claude Opus 4.6"]:
        local_context = get_local_context(directory_path, skip_files=[f for f in os.listdir(directory_path) if f.startswith(company_keyword)])
        blk_posts = "\n--- REJECTED POSTS ---\n" + get_local_context(blocked_path, skip_files=[]) if os.path.exists(blocked_path) else ""

    if model_choice == "Gemini 3.1 Pro":
        run_gemini_briefing_workflow(client_name, company_keyword, google_output_filepath, base_files, blocked_files)
        final_output_filepath = google_output_filepath
    elif model_choice == "GPT-5":
        run_gpt5_briefing_workflow(client_name, company_keyword, gpt_output_filepath, local_context, blk_posts)
        final_output_filepath = gpt_output_filepath
    elif model_choice == "Claude Opus 4.6":
        run_claude_briefing_workflow(client_name, company_keyword, claude_output_filepath, local_context, blk_posts)
        final_output_filepath = claude_output_filepath
    else:
        print(f"Triggering Parallel Briefing Ensemble for {client_name}...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_gemini = executor.submit(run_gemini_briefing_workflow, client_name, company_keyword, google_output_filepath, base_files, blocked_files)
            f_gpt = executor.submit(run_gpt5_briefing_workflow, client_name, company_keyword, gpt_output_filepath, local_context, blk_posts)
            f_claude = executor.submit(run_claude_briefing_workflow, client_name, company_keyword, claude_output_filepath, local_context, blk_posts)
            
            f_gemini.result()
            f_gpt.result()
            f_claude.result()
            
        try:
            with open(google_output_filepath, "r", encoding="utf-8") as f: d_gemini = f.read()
            with open(gpt_output_filepath, "r", encoding="utf-8") as f: d_gpt = f.read()
            with open(claude_output_filepath, "r", encoding="utf-8") as f: d_claude = f.read()
        except Exception as e:
            print(f"File read error during synthesis: {e}")
            d_gemini, d_gpt, d_claude = "", "", ""

        print("Synthesis: Merging insights into Master Briefing...")
        synthesis_prompt = f"""
        You are the Senior Editor-in-Chief. Synthesize these 3 drafts for {client_name} into one Master Briefing.
        Ensure tactical depth, avoid generic AI language, and prioritize the most unique stories from the transcripts.
        Make sure the Domain Knowledge Primer remains fully intact as Section 3.
        Make sure the Dynamic Interview Script (Conversation Tree) is merged into a cohesive, highly usable Section 4.

        DRAFT 1 (Gemini): {d_gemini}
        DRAFT 2 (GPT-5): {d_gpt}
        DRAFT 3 (Claude Draft): {d_claude}
        """
        final_response = anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        final_text = final_response.content[0].text

        with open(final_output_filepath, "w", encoding="utf-8") as out_file:
            out_file.write(f"# MASTER BRIEFING: {client_name.upper()}\n")
            out_file.write("="*60 + "\n\n")
            out_file.write(final_text)

    print(f"\nBriefing complete! Please check the output at: {final_output_filepath}")