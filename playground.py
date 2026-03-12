import os
import time
import json
from google import genai
from google.genai import types

client = genai.Client()

def upload_and_wait(directory, client, skip_files):
    """Helper function to upload files from a directory and wait for processing."""
    uploaded_files = []
    if not os.path.exists(directory):
        return uploaded_files
        
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
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

def generate_iterative_linkedin_posts(client_name, company_keyword):
    directory_path = f"./client_data/{company_keyword}"
    accepted_path = os.path.join(directory_path, "accepted")
    blocked_path = os.path.join(directory_path, "blocked")
    
    output_filename = f"{company_keyword}_generated_posts.txt"
    output_filepath = os.path.join(directory_path, output_filename)
    briefing_filename = f"{company_keyword}_ghostwriter_briefing.md"

    print(f"Scanning directories for {client_name}...")
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    # 1. Upload Base Files (Transcripts & Profile)
    base_files = upload_and_wait(directory_path, client, skip_files=[output_filename, briefing_filename])
    
    # 2. Upload Accepted & Blocked Posts
    accepted_files = upload_and_wait(accepted_path, client, skip_files=[])
    blocked_files = upload_and_wait(blocked_path, client, skip_files=[])
    
    all_uploaded_files = base_files + accepted_files + blocked_files

    print("Files ready. Initializing 3.1 Pro Chat Session...")
    chat = client.chats.create(
        model="gemini-3.1-pro",
        config=types.GenerateContentConfig(temperature=0.7)
    )

    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"VIRIO LINKEDIN POSTS: {client_name.upper()}\n")
        out_file.write("="*50 + "\n\n")

        # --- STEP 1: Context Ingestion ---
        print("\nStep 1: Feeding base context...")
        prompt_1 = f"""
        I am attaching foundational files: interview transcripts between Virio and {client_name}, and their LinkedIn profile. 
        Comprehend these thoroughly to understand their background and product. Do not generate posts yet.
        """
        response_1 = chat.send_message([prompt_1] + base_files)
        out_file.write(f"--- STEP 1: BASE CONTEXT ---\n{response_1.text}\n\n")

        # --- STEP 1.1: Accepted Posts Analysis ---
        if accepted_files:
            print("Step 1.1: Analyzing Accepted Posts...")
            prompt_1a = """
            I am now attaching files containing LinkedIn posts that this client has previously APPROVED and loved.
            Thoroughly analyze these approved posts. What specific tone, formatting choices, sentence structures, and overarching themes does the client prefer?
            """
            response_1a = chat.send_message([prompt_1a] + accepted_files)
            out_file.write(f"--- STEP 1.1: APPROVED POSTS ANALYSIS ---\n{response_1a.text}\n\n")

        # --- STEP 1.2: Blocked Posts Analysis ---
        if blocked_files:
            print("Step 1.2: Analyzing Rejected Posts...")
            prompt_1b = """
            I am now attaching files containing LinkedIn posts that this client has REJECTED.
            Thoroughly analyze these rejected posts. Contrast them with the approved ones. What specific mistakes, tones, cliches, or formatting choices must we strictly AVOID moving forward?
            """
            response_1b = chat.send_message([prompt_1b] + blocked_files)
            out_file.write(f"--- STEP 1.2: REJECTED POSTS ANALYSIS (AVOID) ---\n{response_1b.text}\n\n")

        # --- STEP 2: ICP & Product Analysis ---
        print("Step 2: Analyzing ICP...")
        prompt_2 = f"Based on everything so far, what is {client_name}'s ideal customer profile? What is their product, and how does it solve the ICP's pain points?"
        response_2 = chat.send_message(prompt_2)
        out_file.write(f"--- STEP 2: ICP & PRODUCT ANALYSIS ---\n{response_2.text}\n\n")

        # --- STEP 3: Overarching Strategy ---
        print("Step 3: Developing 10 overarching messages...")
        prompt_3 = f"""
        We need 10 overarching messages that range from BOFU to MOFU to TOFU. 
        Leverage {client_name}'s philosophies. Ensure these messages align with what we learned from their "Approved" posts and explicitly avoid the pitfalls of their "Rejected" posts.
        
        CRITICAL INSTRUCTION: Output your response EXCLUSIVELY as a valid JSON array of 10 strings. 
        Do not include markdown blocks, numbering, or conversational text.
        """
        response_3 = chat.send_message(prompt_3)
        
        try:
            clean_json = response_3.text.replace("```json", "").replace("```", "").strip()
            messages_list = json.loads(clean_json)
            out_file.write("--- STEP 3: THE 10 OVERARCHING MESSAGES ---\n")
            for i, msg in enumerate(messages_list):
                out_file.write(f"{i+1}. {msg}\n")
            out_file.write("\n")
        except json.JSONDecodeError:
            print("Error parsing JSON. Exiting.")
            return

        # --- STEP 4: Example Comprehension ---
        print("Step 4: Comprehending Example Posts...")
        prompt_4 = """
        Before we write, read these perfect baseline examples from Virio's internal style guide:
        
        [PASTE YOUR LONG VIRIO EXAMPLES HERE]
        
        Acknowledge what makes these succinct, snappy, and effective.
        """
        response_4 = chat.send_message(prompt_4)
        out_file.write(f"--- STEP 4: VIRIO STYLE GUIDE ANALYSIS ---\n{response_4.text}\n\n")

        # --- STEP 5: Iterative Generation ---
        print("Step 5: Drafting the 10 LinkedIn posts iteratively...")
        out_file.write("="*50 + "\n--- FINAL LINKEDIN POST DRAFTS ---\n" + "="*50 + "\n\n")
        
        for index, message in enumerate(messages_list):
            print(f"Generating Post {index + 1} of 10...")
            prompt_5 = f"""
            Theme for this post: "{message}"
            
            Generate a LinkedIn post rooted in {client_name}'s background.
            CRITICAL: You must merge the snappy structure of the Virio examples with the specific tone and preferences we identified in the client's APPROVED posts. 
            Absolutely do NOT use the tones or styles identified in the REJECTED posts.
            """
            response_5 = chat.send_message(prompt_5)
            
            out_file.write(f"POST {index + 1} THEME: {message}\n")
            out_file.write("-" * 25 + "\n")
            out_file.write(f"{response_5.text}\n\n")
            out_file.write("*" * 50 + "\n\n")

    print("\nCleaning up files from servers...")
    for f in all_uploaded_files:
        client.files.delete(name=f.name)
        
    print(f"Process complete! Output saved to: {output_filepath}")