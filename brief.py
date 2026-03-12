import os
import time
import docx
import concurrent.futures
from google import genai
from google.genai import types
from anthropic import Anthropic
from openai import OpenAI

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
        
        # Only read .txt files! The .docx files are already converted by upload_and_wait.
        if filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                context_text += f"\n--- DOCUMENT: {filename} ---\n{f.read()}\n"
                
    return context_text

def upload_and_wait(directory, client, skip_files):
    """Helper function to convert docs, upload files, and wait for processing."""
    uploaded_files = []
    if not os.path.exists(directory):
        return uploaded_files
        
    # 1. Pre-process: Find any .docx files and convert them to .txt
    for filename in os.listdir(directory):
        if filename.lower().endswith(".docx"):
            filepath = os.path.join(directory, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_filepath = os.path.join(directory, txt_filename)
            
            # Only convert if we haven't already converted it previously
            if not os.path.exists(txt_filepath):
                print(f"Converting {filename} to TXT...")
                doc = docx.Document(filepath)
                with open(txt_filepath, "w", encoding="utf-8") as f:
                    f.write("\n".join([p.text for p in doc.paragraphs]))
    
    # 2. Upload all supported files
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip the original .docx files (we will upload the .txt versions instead)
        if filename.lower().endswith(".docx"):
            continue
            
        if os.path.isfile(filepath) and filename not in skip_files:
            print(f"Uploading {filename} from {os.path.basename(directory)}...")
            f = client.files.upload(file=filepath)
            uploaded_files.append(f)
            
    # 3. Wait for Google's servers to process the uploads
    for i, f in enumerate(uploaded_files):
        while f.state.name == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        uploaded_files[i] = f
        
    return uploaded_files

def run_gemini_workflow(client_name, output_filepath, base_files, accepted_posts, blocked_posts):

    print("Files ready. Initializing 3.1 Pro Chat Session...")
    chat = google_client.chats.create(
        model="gemini-3.1-pro-preview",
        config=types.GenerateContentConfig(temperature=0.5)
    )

    # Initialize a string to capture the full output
    full_briefing_content = ""

    try:

        with open(output_filepath, "w", encoding="utf-8") as out_file:
            header = f"#Briefing: {client_name.upper()}\n\n"
            out_file.write(header)
            full_briefing_content += header

            # --- STEP 1: Context Ingestion ---
            print("\nGemini Step 1: Feeding context...")
            prompt_1 = f"""
            I am attaching several files: a PDF export of the LinkedIn profile for {client_name}, 
            and transcripts of past interviews conducted between our content agency (Virio) and them.
            
            Your task is to help me build a comprehensive "Ghostwriter Briefing Document" step-by-step. 
            This document will be handed to a new ghostwriter who has zero prior knowledge of the client. 
            
            First, acknowledge that you have read and comprehended the files. Do not generate the briefing yet.
            """
            chat.send_message([prompt_1] + base_files)

            # --- STEP 1.1: Accepted Posts Analysis ---
            if accepted_posts:
                print("Gemini Step 1.1: Processing Accepted Posts...")
                prompt_1a = """
                I am attaching LinkedIn posts this client has APPROVED. 
                Analyze them to understand the specific persona, topics, and angles the client likes to project publicly. 
                Acknowledge receipt and summarize their preferred persona.
                """
                chat.send_message([prompt_1a] + accepted_posts)

            # --- STEP 1.2: Blocked Posts Analysis ---
            if blocked_posts:
                print("Gemini Step 1.2: Processing Rejected Posts...")
                prompt_1b = """
                I am attaching LinkedIn posts this client has REJECTED. 
                Analyze them to understand what topics, tones, or angles we must avoid. 
                Acknowledge receipt and summarize the "Do Not Cross" boundaries for this client.
                """
                chat.send_message([prompt_1b] + blocked_posts)

            # --- STEP 2: The Client & ICP ---
            print("Gemini Step 2: Drafting Client Background & ICP...")
            prompt_2 = f"""
            Let's write the first two sections of the document. Use Markdown formatting.
            
            ## 1. The Client
            Provide a crystal-clear breakdown of {client_name}'s professional background, notable achievements, and current role.
            
            ## 2. The Ideal Customer Profile (ICP)
            Who exactly are they selling to? What are the ICP's pain points, desires, and psychological triggers?
            """
            response_2 = chat.send_message(prompt_2)
            section_2 = response_2.text + "\n\n"
            out_file.write(section_2)
            full_briefing_content += section_2

            # --- STEP 3: Content Strategy ---
            print("Gemini Step 3: Drafting Content Strategy...")
            prompt_3 = f"""
            Now, generate the third section:
            
            ## 3. Content Strategy & Persona
            How does {client_name} want to be perceived by their ICP? What is their unique philosophy, and how does their product/service naturally position itself as the solution?
            Crucially, include a "Green Flags" (what they love) and "Red Flags" (what to strictly avoid based on rejected posts) subsection for the ghostwriter to keep in mind.
            """
            response_3 = chat.send_message(prompt_3)
            section_3 = response_3.text + "\n\n"
            out_file.write(section_3)
            full_briefing_content += section_3

            # --- STEP 4: Previously Covered Ground ---
            print("Gemini Step 4: Summarizing Past Interviews...")
            prompt_4 = """
            Next, generate the fourth section based strictly on the interview transcripts:
            
            ## 4. Previously Covered Ground
            Provide thorough, structured summaries of the attached past interviews. What core stories, frameworks, and philosophies have we already extracted from them?
            """
            response_4 = chat.send_message(prompt_4)
            section_4 = response_4.text + "\n\n"
            out_file.write(section_4)
            full_briefing_content += section_4

            # --- STEP 5: Next Interview Strategy ---
            print("Gemini Step 5: Drafting the Next Interview Playbook...")
            prompt_5 = """
            Finally, generate the most critical section for our new ghostwriter:
            
            ## 5. Next Interview Strategy (The Playbook)
            Based on what we already know (and what we are missing from the transcripts), provide a strict strategy for the upcoming 1-hour interview. 
            - Include the overarching goal of the next call.
            - Provide 9-13 highly specific, probing questions the ghostwriter must ask to mine new, unseen stories or deeper tactical insights. 
            - Include elaborate notes on how to steer the client if they start giving generic or "corporate" answers.
            """
            response_5 = chat.send_message(prompt_5)
            section_5 = response_5.text + "\n\n"
            out_file.write(section_5)
            full_briefing_content += section_5

    finally:
        print("\nCleaning up files from Google servers...")
        for f in base_files + accepted_posts + blocked_posts:
            try:
                google_client.files.delete(name=f.name)
            except Exception as e:
                print(f"Failed to delete {f.name}: {e}")
            
    print(f"Process complete! Sectioned Briefing saved to: {output_filepath}")
    return full_briefing_content

def run_gpt5_workflow(client_name, output_filepath, context, accepted_posts, blocked_posts):
    """Executes the full prompt sequence in one pass for GPT-5."""
    print("\nFeeding GPT context...")
    full_prompt = f"""
        I am attaching several files: a PDF export of the LinkedIn profile for {client_name}, 
        and transcripts of past interviews conducted between our content agency (Virio) and them.
        
        Your task is to help me build a comprehensive "Ghostwriter Briefing Document" step-by-step. 
        This document will be handed to a new ghostwriter who has zero prior knowledge of the client. 
        
        First, acknowledge that you have read and comprehended the files. Do not generate the briefing yet.

        Client: {client_name}
        LinkedIn PDF export + Transcripts: {context}

        I am attaching LinkedIn posts this client has APPROVED. 
        Analyze them to understand the specific persona, topics, and angles the client likes to project publicly. 
        Acknowledge receipt and summarize their preferred persona.

        ACCEPTED POSTS: {accepted_posts}

        I am attaching LinkedIn posts this client has REJECTED. 
        Analyze them to understand what topics, tones, or angles we must avoid. 
        Acknowledge receipt and summarize the "Do Not Cross" boundaries for this client.

        REJECTED POSTS: {blocked_posts}

        Let's write the first two sections of the document. Use Markdown formatting.
        
        ## 1. The Client
        Provide a crystal-clear breakdown of {client_name}'s professional background, notable achievements, and current role.
        
        ## 2. The Ideal Customer Profile (ICP)
        Who exactly are they selling to? What are the ICP's pain points, desires, and psychological triggers?

        Now, generate the third section:
        
        ## 3. Content Strategy & Persona
        How does {client_name} want to be perceived by their ICP? What is their unique philosophy, and how does their product/service naturally position itself as the solution?
        Crucially, include a "Green Flags" (what they love) and "Red Flags" (what to strictly avoid based on rejected posts) subsection for the ghostwriter to keep in mind.

        Next, generate the fourth section based strictly on the interview transcripts:
        
        ## 4. Previously Covered Ground
        Provide thorough, structured summaries of the attached past interviews. What core stories, frameworks, and philosophies have we already extracted from them?

        Finally, generate the most critical section for our new ghostwriter:
        
        ## 5. Next Interview Strategy (The Playbook)
        Based on what we already know (and what we are missing from the transcripts), provide a strict strategy for the upcoming 1-hour interview. 
        - Include the overarching goal of the next call.
        - Provide 9-13 highly specific, probing questions the ghostwriter must ask to mine new, unseen stories or deeper tactical insights. 
        - Include elaborate notes on how to steer the client if they start giving generic or "corporate" answers.
    """
    response = openai_client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "system", "content": "You are a professional ghostwriting strategist."},
                {"role": "user", "content": full_prompt}]
    )
    
    # Extract the generated text
    gpt_content = response.choices[0].message.content
    
    # Write the draft to its own file
    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"# GPT-5 Draft Briefing: {client_name.upper()}\n\n")
        out_file.write(gpt_content)
        
    print(f"GPT-5 draft saved to: {output_filepath}")
    return gpt_content

def run_claude_workflow(client_name, output_filepath, context, accepted_posts, blocked_posts):
    """Executes the full prompt sequence in one pass for Claude 4 Opus."""
    print("\nFeeding Claude context...")
    full_prompt = f"""
        I am attaching several files: a PDF export of the LinkedIn profile for {client_name}, 
        and transcripts of past interviews conducted between our content agency (Virio) and them.
        
        Your task is to help me build a comprehensive "Ghostwriter Briefing Document" step-by-step. 
        This document will be handed to a new ghostwriter who has zero prior knowledge of the client. 
        
        First, acknowledge that you have read and comprehended the files. Do not generate the briefing yet.

        Client: {client_name}
        LinkedIn PDF export + Transcripts: {context}

        I am attaching any LinkedIn posts this client has APPROVED. 
        Analyze them to understand the specific persona, topics, and angles the client likes to project publicly. 
        Acknowledge receipt and summarize their preferred persona.

        ACCEPTED POSTS: {accepted_posts}

        I am attaching any LinkedIn posts this client has REJECTED. 
        Analyze them to understand what topics, tones, or angles we must avoid. 
        Acknowledge receipt and summarize the "Do Not Cross" boundaries for this client.

        REJECTED POSTS: {blocked_posts}

        Let's write the first two sections of the document. Use Markdown formatting.
        
        ## 1. The Client
        Provide a crystal-clear breakdown of {client_name}'s professional background, notable achievements, and current role.
        
        ## 2. The Ideal Customer Profile (ICP)
        Who exactly are they selling to? What are the ICP's pain points, desires, and psychological triggers?

        Now, generate the third section:
        
        ## 3. Content Strategy & Persona
        How does {client_name} want to be perceived by their ICP? What is their unique philosophy, and how does their product/service naturally position itself as the solution?
        Crucially, include a "Green Flags" (what they love) and "Red Flags" (what to strictly avoid based on rejected posts) subsection for the ghostwriter to keep in mind.

        Next, generate the fourth section based strictly on the interview transcripts:
        
        ## 4. Previously Covered Ground
        Provide thorough, structured summaries of the attached past interviews. What core stories, frameworks, and philosophies have we already extracted from them?

        Finally, generate the most critical section for our new ghostwriter:
        
        ## 5. Next Interview Strategy (The Playbook)
        Based on what we already know (and what we are missing from the transcripts), provide a strict strategy for the upcoming 1-hour interview. 
        - Include the overarching goal of the next call.
        - Provide 9-13 highly specific, probing questions the ghostwriter must ask to mine new, unseen stories or deeper tactical insights. 
        - Include elaborate notes on how to steer the client if they start giving generic or "corporate" answers.
    """
    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8196,
        messages=[{"role": "user", "content": full_prompt}]
    )
    
    # Extract the generated text
    claude_content = response.content[0].text
    
    # Write the draft to its own file
    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"# Claude Draft Briefing: {client_name.upper()}\n\n")
        out_file.write(claude_content)
        
    print(f"Claude draft saved to: {output_filepath}")
    return claude_content

# --- MAIN ORCHESTRATOR ---

def generate_briefing(client_name, company_keyword, model_choice="All (Ensemble)"):

    directory_path = f"./client_data/{company_keyword}"
    accepted_path = os.path.join(directory_path, "accepted")
    blocked_path = os.path.join(directory_path, "rejected")
    
    post_filename = f"{company_keyword}_ruanmei_posts.txt"
    google_output_filename = f"{company_keyword}_ruanmei_gemini_briefing.md"
    google_output_filepath = os.path.join(directory_path, google_output_filename)
    gpt_output_filename = f"{company_keyword}_ruanmei_gpt_briefing.md"
    gpt_output_filepath = os.path.join(directory_path, gpt_output_filename)
    claude_output_filename = f"{company_keyword}_ruanmei_claude_briefing.md"
    claude_output_filepath = os.path.join(directory_path, claude_output_filename)
    
    # The viewer always looks for this specific file, so we map our final output here
    final_output_filename = f"{company_keyword}_ruanmei_briefing.md"
    final_output_filepath = os.path.join(directory_path, final_output_filename)

    print(f"Scanning directories for {client_name}...")
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    # 1. Prep context for Gemini (File API) - Only run if Gemini or Ensemble is selected
    if model_choice in ["All (Ensemble)", "Gemini 3.1 Pro"]:
        base_files = upload_and_wait(directory_path, google_client, skip_files=[post_filename, google_output_filename, gpt_output_filename, claude_output_filename, final_output_filename])
        acc_files = upload_and_wait(accepted_path, google_client, skip_files=[])
        blk_files = upload_and_wait(blocked_path, google_client, skip_files=[])
    
    # 2. Prep context for GPT/Claude (Raw Text) - Only run if GPT, Claude, or Ensemble is selected
    if model_choice in ["All (Ensemble)", "GPT-5", "Claude Opus 4.6"]:
        local_context = get_local_context(directory_path, skip_files=[post_filename, google_output_filename, gpt_output_filename, claude_output_filename, final_output_filename])
        acc_posts = "\n--- APPROVED POSTS ---\n" + get_local_context(accepted_path, skip_files=[])
        blk_posts = "\n--- REJECTED POSTS ---\n" + get_local_context(blocked_path, skip_files=[])

    # 3. Execution Routing based on user choice
    if model_choice == "Gemini 3.1 Pro":
        print(f"Running Solo Briefing with Gemini for {client_name}...")
        final_text = run_gemini_workflow(client_name, google_output_filepath, base_files, acc_files, blk_files)
        
    elif model_choice == "GPT-5":
        print(f"Running Solo Briefing with GPT-5 for {client_name}...")
        final_text = run_gpt5_workflow(client_name, gpt_output_filepath, local_context, acc_posts, blk_posts)
        
    elif model_choice == "Claude Opus 4.6":
        print(f"Running Solo Briefing with Claude for {client_name}...")
        final_text = run_claude_workflow(client_name, claude_output_filepath, local_context, acc_posts, blk_posts)
        
    else:
        # Full Ensemble Mode
        print(f"Triggering Parallel Briefing Ensemble for {client_name}...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_gemini = executor.submit(run_gemini_workflow, client_name, google_output_filepath, base_files, acc_files, blk_files)
            f_gpt = executor.submit(run_gpt5_workflow, client_name, gpt_output_filepath, local_context, acc_posts, blk_posts)
            f_claude = executor.submit(run_claude_workflow, client_name, claude_output_filepath, local_context, acc_posts, blk_posts)

            d_gemini = f_gemini.result()
            d_gpt = f_gpt.result()
            d_claude = f_claude.result()

        print("Synthesis: Merging insights into Master Briefing...")
        synthesis_prompt = f"""
        You are the Senior Editor-in-Chief. Synthesize these 3 drafts for {client_name} into one Master Briefing.
        Ensure tactical depth, avoid generic AI language, and prioritize the most unique stories from the transcripts.

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

    # 4. Save the Final Output
    match model_choice:
        case "Gemini 3.1 Pro":
            final_output_filepath = google_output_filepath
        case "GPT-5":
            final_output_filepath = gpt_output_filepath
        case "Claude Opus 4.6":
            final_output_filepath = claude_output_filepath
    with open(final_output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"# MASTER BRIEFING: {client_name.upper()}\n")
        out_file.write(f"*(Generated via: {model_choice})*\n\n")
        out_file.write(final_text)
        
    print(f"Process complete! Briefing saved to: {final_output_filepath}")