import os
import time
import docx
from google import genai
from google.genai import types

client = genai.Client()

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

def generate_ghostwriter_briefing(client_name, company_keyword):
    directory_path = f"./client_data/{company_keyword}"
    accepted_path = os.path.join(directory_path, "accepted")
    blocked_path = os.path.join(directory_path, "rejected")
    
    post_filename = f"{company_keyword}_ruanmei_posts.txt"
    output_filename = f"{company_keyword}_ruanmei_briefing.md"
    output_filepath = os.path.join(directory_path, output_filename)

    print(f"Scanning directories for {client_name}...")
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    # 1. Upload Base Files (Transcripts & Profile)
    base_files = upload_and_wait(directory_path, client, skip_files=[post_filename, output_filename])
    
    # 2. Upload Accepted & Blocked Posts
    accepted_files = upload_and_wait(accepted_path, client, skip_files=[])
    blocked_files = upload_and_wait(blocked_path, client, skip_files=[])
    
    all_uploaded_files = base_files + accepted_files + blocked_files

    print("Files ready. Initializing 3.1 Pro Chat Session...")
    chat = client.chats.create(
        model="gemini-3.1-pro-preview",
        config=types.GenerateContentConfig(temperature=0.5)
    )

    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"#Briefing: {client_name.upper()}\n\n")

        # --- STEP 1: Context Ingestion ---
        print("\nStep 1: Feeding context...")
        prompt_1 = f"""
        I am attaching several files: a PDF export of the LinkedIn profile for {client_name}, 
        and transcripts of past interviews conducted between our content agency (Virio) and them.
        
        Your task is to help me build a comprehensive "Ghostwriter Briefing Document" step-by-step. 
        This document will be handed to a new ghostwriter who has zero prior knowledge of the client. 
        
        First, acknowledge that you have read and comprehended the files. Do not generate the briefing yet.
        """
        chat.send_message([prompt_1] + base_files)

        # --- STEP 1.1: Accepted Posts Analysis ---
        if accepted_files:
            print("Step 1.1: Processing Accepted Posts...")
            prompt_1a = """
            I am attaching LinkedIn posts this client has APPROVED. 
            Analyze them to understand the specific persona, topics, and angles the client likes to project publicly. 
            Acknowledge receipt and summarize their preferred persona.
            """
            chat.send_message([prompt_1a] + accepted_files)

        # --- STEP 1.2: Blocked Posts Analysis ---
        if blocked_files:
            print("Step 1.2: Processing Rejected Posts...")
            prompt_1b = """
            I am attaching LinkedIn posts this client has REJECTED. 
            Analyze them to understand what topics, tones, or angles we must avoid. 
            Acknowledge receipt and summarize the "Do Not Cross" boundaries for this client.
            """
            chat.send_message([prompt_1b] + blocked_files)

        # --- STEP 2: The Client & ICP ---
        print("Step 2: Drafting Client Background & ICP...")
        prompt_2 = f"""
        Let's write the first two sections of the document. Use Markdown formatting.
        
        ## 1. The Client
        Provide a crystal-clear breakdown of {client_name}'s professional background, notable achievements, and current role.
        
        ## 2. The Ideal Customer Profile (ICP)
        Who exactly are they selling to? What are the ICP's pain points, desires, and psychological triggers?
        """
        response_2 = chat.send_message(prompt_2)
        out_file.write(response_2.text + "\n\n")

        # --- STEP 3: Content Strategy ---
        print("Step 3: Drafting Content Strategy...")
        prompt_3 = f"""
        Now, generate the third section:
        
        ## 3. Content Strategy & Persona
        How does {client_name} want to be perceived by their ICP? What is their unique philosophy, and how does their product/service naturally position itself as the solution?
        Crucially, include a "Green Flags" (what they love) and "Red Flags" (what to strictly avoid based on rejected posts) subsection for the ghostwriter to keep in mind.
        """
        response_3 = chat.send_message(prompt_3)
        out_file.write(response_3.text + "\n\n")

        # --- STEP 4: Previously Covered Ground ---
        print("Step 4: Summarizing Past Interviews...")
        prompt_4 = """
        Next, generate the fourth section based strictly on the interview transcripts:
        
        ## 4. Previously Covered Ground
        Provide thorough, structured summaries of the attached past interviews. What core stories, frameworks, and philosophies have we already extracted from them?
        """
        response_4 = chat.send_message(prompt_4)
        out_file.write(response_4.text + "\n\n")

        # --- STEP 5: Next Interview Strategy ---
        print("Step 5: Drafting the Next Interview Playbook...")
        prompt_5 = """
        Finally, generate the most critical section for our new ghostwriter:
        
        ## 5. Next Interview Strategy (The Playbook)
        Based on what we already know (and what we are missing from the transcripts), provide a strict strategy for the upcoming 1-hour interview. 
        - Include the overarching goal of the next call.
        - Provide 9-13 highly specific, probing questions the ghostwriter must ask to mine new, unseen stories or deeper tactical insights. 
        - Include elaborate notes on how to steer the client if they start giving generic or "corporate" answers.
        """
        response_5 = chat.send_message(prompt_5)
        out_file.write(response_5.text + "\n\n")

    print("\nCleaning up files from servers...")
    for f in all_uploaded_files:
        client.files.delete(name=f.name)
        
    print(f"Process complete! Sectioned Briefing saved to: {output_filepath}")