import os
import time
import docx
import json
import concurrent.futures
from google import genai
from google.genai import types
from anthropic import Anthropic
from openai import OpenAI

from cyrene import Cyrene

google_client = genai.Client()
anthropic_client = Anthropic()
openai_client = OpenAI()

# immediate next steps
# adjustable hook and tone (hooks are the most important)
# reduce from dense paragraph-form 
# perhaps ensemble of writers that each adopt a distinct tone
# tones can be gather from predetermined set

# proactive behavior
# distillation, tune? 

def get_local_context(directory, skip_files):
    """Extracts raw text from local files for non-Google models."""
    context_text = ""
    if not os.path.exists(directory):
        return context_text
    for filename in os.listdir(directory):
        if filename in skip_files: continue
        filepath = os.path.join(directory, filename)
        
        # Only read .txt files and .md files (for ABM profiles)
        if filename.lower().endswith((".txt", ".md")):
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

def run_gemini_posts_workflow(client_name, output_filepath, base_files, accepted_files, blocked_files, abm_files):
    print("Files ready. Initializing Gemini Chat Session...")
    all_uploaded_files = base_files + accepted_files + blocked_files + abm_files
    
    chat = google_client.chats.create(
        model="gemini-3.1-pro",
        config=types.GenerateContentConfig(temperature=0.7)
    )

    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"RUAN MEI LINKEDIN POSTS (GEMINI): {client_name.upper()}\n")
        out_file.write("="*50 + "\n\n")

        # --- STEP 1: Context Ingestion ---
        print("\nGemini Step 1: Feeding context...")
        prompt_1 = f"""
        I am attaching several files. Among these are interview transcripts between our organization Virio (an organization that writes LinkedIn posts for startup founders and executives)
        and a startup founder/executive named {client_name}, as well as a PDF export of their LinkedIn profile. 
        Comprehend the LinkedIn profile and these interview transcripts thoroughly before we proceed.
        """
        response_1 = chat.send_message([prompt_1] + base_files)
        out_file.write(f"--- STEP 1: CONTEXT INGESTION ---\n{response_1.text}\n\n")

        # --- STEP 1.1: Accepted Posts Analysis ---
        if accepted_files:
            print("Gemini Step 1.1: Analyzing Accepted Posts...")
            prompt_1a = """
            I am now attaching files containing LinkedIn posts that this client has previously APPROVED and loved.
            Thoroughly analyze these approved posts. What specific formatting choices, sentence length/structures, tone, and overarching themes does the client prefer?
            """
            response_1a = chat.send_message([prompt_1a] + accepted_files)
            out_file.write(f"--- STEP 1.1: APPROVED POSTS ANALYSIS ---\n{response_1a.text}\n\n")

        # --- STEP 1.2: Blocked Posts Analysis ---
        if blocked_files:
            print("Gemini Step 1.2: Analyzing Rejected Posts...")
            prompt_1b = """
            I am now attaching files containing LinkedIn posts that this client has REJECTED.
            Thoroughly analyze these rejected posts. Contrast them with the approved ones. What specific content inaccuracies, tones, cliches, or formatting choices must we strictly AVOID moving forward?
            """
            response_1b = chat.send_message([prompt_1b] + blocked_files)
            out_file.write(f"--- STEP 1.2: REJECTED POSTS ANALYSIS (AVOID) ---\n{response_1b.text}\n\n")

        # --- STEP 2: ICP & Product Analysis ---
        print("Gemini Step 2: Analyzing ICP...")
        prompt_2 = f"First, what is {client_name}'s ideal customer profile? What is his/her product? How would his/her product and philosophies resonate with his/her ideal customer?"
        response_2 = chat.send_message(prompt_2)
        out_file.write(f"--- STEP 2: ICP & PRODUCT ANALYSIS ---\n{response_2.text}\n\n")

        # --- STEP 2.5: ABM Targets ---
        abm_instruction = ""
        if abm_files:
            print("Gemini Step 2.5: Ingesting ABM Profiles...")
            prompt_2_5 = "I am attaching Account-Based Marketing (ABM) target profiles. Please review these targets, their predicted pain points, and recommended ingress strategies."
            response_2_5 = chat.send_message([prompt_2_5] + abm_files)
            out_file.write(f"--- STEP 2.5: ABM PROFILES INGESTION ---\n{response_2_5.text}\n\n")
            abm_instruction = "CRITICAL: Exactly 3 of these 12 messages MUST be Account-Based Marketing (ABM) posts dedicated to the specific targets provided in the ABM profiles. These ABM messages should favorably mention the target, address their specific pain points, and subtly position {client_name} as the solution to encourage a meeting."

        # --- STEP 3: Overarching Strategy ---
        print("Gemini Step 3: Developing 12 overarching messages...")
        prompt_3 = f"""
        We need 12 posts that range from BOFU to MOFU to TOFU. 
        Begin with drafting 12 compelling overarching messages that you would like our posts to deliver. 
        Let's leverage a variety of {client_name}'s practices and philosophies to appeal to {client_name}'s ICP.
        {abm_instruction}
        CRITICAL INSTRUCTION: Output your response EXCLUSIVELY as a valid JSON array of 12 strings. 
        """
        response_3 = chat.send_message(prompt_3)
        
        try:
            clean_json = response_3.text.replace("```json", "").replace("```", "").strip()
            if "[" in clean_json and "]" in clean_json:
                clean_json = clean_json[clean_json.find("["):clean_json.rfind("]")+1]
            messages_list = json.loads(clean_json)
            out_file.write("--- STEP 3: THE 12 OVERARCHING MESSAGES ---\n")
            for i, msg in enumerate(messages_list):
                out_file.write(f"{i+1}. {msg}\n")
            out_file.write("\n")
        except json.JSONDecodeError:
            print("Error parsing JSON. Exiting Gemini thread.")
            return

        # --- STEP 4: Iterative Generation ---
        print("Gemini Step 4: Drafting the 12 LinkedIn posts iteratively...")
        out_file.write("="*50 + "\n--- FINAL LINKEDIN POST DRAFTS ---\n" + "="*50 + "\n\n")
        
        for index, message in enumerate(messages_list):
            print(f"Generating Post {index + 1} of 12...")
            prompt_5 = f"""
            Theme for this post: "{message}"
            
            Generate a LinkedIn post around this theme that delivers practical advice and/or example(s) throughout the post, culminating 
            in a succinct conclusion. INCORPORATE ELEMENTS FROM APPROVED POSTS AND AVOID ELEMENTS FROM REJECTED POSTS.
            
            If this theme mentions an ABM target, strictly follow the ingress strategy from their profile, mention them favorably, and structure the post to naturally encourage them to reach out or accept a meeting with {client_name}.
            
            The example should be rooted in {client_name}'s real or PLAUSIBLY real experiences.
            Ensure you strictly apply a snappy, succinct writing style as we analyzed in the previous step.
            """
            response_5 = chat.send_message(prompt_5)
            
            out_file.write(f"POST {index + 1} THEME: {message}\n")
            out_file.write("-" * 25 + "\n")
            out_file.write(f"{response_5.text}\n\n")
            out_file.write("*" * 50 + "\n\n")

    print("\nCleaning up files from Google servers...")
    for f in all_uploaded_files:
        try:
            google_client.files.delete(name=f.name)
        except Exception as e:
            print(f"Failed to delete {f.name}: {e}")

def run_gpt5_posts_workflow(client_name, output_filepath, base_text, acc_text, blk_text, abm_text):
    print("Initializing GPT-5 Chat Session...")
    messages = [{"role": "system", "content": "You are a professional LinkedIn ghostwriter."}]
    
    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"RUAN MEI LINKEDIN POSTS (GPT-5): {client_name.upper()}\n")
        out_file.write("="*50 + "\n\n")

        # --- STEP 1 ---
        print("\nGPT-5 Step 1: Feeding context...")
        prompt_1 = f"""
        I am attaching several files. Among these are interview transcripts between our organization Virio (an organization that writes LinkedIn posts for startup founders and executives)
        and a startup founder/executive named {client_name}, as well as a PDF export of their LinkedIn profile. 
        Comprehend the LinkedIn profile and these interview transcripts thoroughly before we proceed.
        """
        messages.append({"role": "user", "content": prompt_1 + f"\n\nFILES:\n{base_text}"})
        resp_1 = openai_client.chat.completions.create(model="gpt-5", messages=messages)
        resp_1_text = resp_1.choices[0].message.content
        messages.append({"role": "assistant", "content": resp_1_text})
        out_file.write(f"--- STEP 1: CONTEXT INGESTION ---\n{resp_1_text}\n\n")

        # --- STEP 1.1 ---
        if acc_text:
            print("GPT-5 Step 1.1: Analyzing Accepted Posts...")
            prompt_1a = """
            I am now attaching files containing LinkedIn posts that this client has previously APPROVED and loved.
            Thoroughly analyze these approved posts. What specific tone, formatting choices, sentence structures, and overarching themes does the client prefer?
            """
            messages.append({"role": "user", "content": prompt_1a + f"\n\nAPPROVED POSTS:\n{acc_text}"})
            resp_1a = openai_client.chat.completions.create(model="gpt-5", messages=messages)
            resp_1a_text = resp_1a.choices[0].message.content
            messages.append({"role": "assistant", "content": resp_1a_text})
            out_file.write(f"--- STEP 1.1: APPROVED POSTS ANALYSIS ---\n{resp_1a_text}\n\n")

        # --- STEP 1.2 ---
        if blk_text:
            print("GPT-5 Step 1.2: Analyzing Rejected Posts...")
            prompt_1b = """
            I am now attaching files containing LinkedIn posts that this client has REJECTED.
            Thoroughly analyze these rejected posts. Contrast them with the approved ones. What specific content inaccuracies, tones, cliches, or formatting choices must we strictly AVOID moving forward?
            """
            messages.append({"role": "user", "content": prompt_1b + f"\n\nREJECTED POSTS:\n{blk_text}"})
            resp_1b = openai_client.chat.completions.create(model="gpt-5", messages=messages)
            resp_1b_text = resp_1b.choices[0].message.content
            messages.append({"role": "assistant", "content": resp_1b_text})
            out_file.write(f"--- STEP 1.2: REJECTED POSTS ANALYSIS (AVOID) ---\n{resp_1b_text}\n\n")

        # --- STEP 2 ---
        print("GPT-5 Step 2: Analyzing ICP...")
        prompt_2 = f"First, what is {client_name}'s ideal customer profile? What is his/her product? How would his/her product and philosophies resonate with his/her ideal customer?"
        messages.append({"role": "user", "content": prompt_2})
        resp_2 = openai_client.chat.completions.create(model="gpt-5", messages=messages)
        resp_2_text = resp_2.choices[0].message.content
        messages.append({"role": "assistant", "content": resp_2_text})
        out_file.write(f"--- STEP 2: ICP & PRODUCT ANALYSIS ---\n{resp_2_text}\n\n")

        # --- STEP 2.5 ---
        abm_instruction = ""
        if abm_text:
            print("GPT-5 Step 2.5: Ingesting ABM Profiles...")
            prompt_2_5 = f"""
            I am attaching Account-Based Marketing (ABM) target profiles. Please review these targets, their predicted pain points, and recommended ingress strategies.
            \n\nABM PROFILES:\n{abm_text}
            """
            messages.append({"role": "user", "content": prompt_2_5})
            resp_2_5 = openai_client.chat.completions.create(model="gpt-5", messages=messages)
            resp_2_5_text = resp_2_5.choices[0].message.content
            messages.append({"role": "assistant", "content": resp_2_5_text})
            out_file.write(f"--- STEP 2.5: ABM PROFILES INGESTION ---\n{resp_2_5_text}\n\n")
            abm_instruction = "CRITICAL: Exactly 3 of these 12 messages MUST be Account-Based Marketing (ABM) posts dedicated to the specific targets provided in the ABM profiles. These ABM messages should favorably mention the target, address their specific pain points, and subtly position {client_name} as the solution to encourage a meeting."

        # --- STEP 3 ---
        print("GPT-5 Step 3: Developing 12 overarching messages...")
        prompt_3 = f"""
        We need 12 posts that range from BOFU to MOFU to TOFU. 
        Begin with drafting 12 compelling overarching messages that you would like our posts to deliver. 
        Let's leverage a variety of {client_name}'s practices and philosophies to appeal to {client_name}'s ICP.
        {abm_instruction}
        CRITICAL INSTRUCTION: Output your response EXCLUSIVELY as a valid JSON array of 12 strings. 
        """
        messages.append({"role": "user", "content": prompt_3})
        resp_3 = openai_client.chat.completions.create(model="gpt-5", messages=messages)
        resp_3_text = resp_3.choices[0].message.content
        messages.append({"role": "assistant", "content": resp_3_text})
        
        try:
            clean_json = resp_3_text.replace("```json", "").replace("```", "").strip()
            if "[" in clean_json and "]" in clean_json:
                clean_json = clean_json[clean_json.find("["):clean_json.rfind("]")+1]
            messages_list = json.loads(clean_json)
            out_file.write("--- STEP 3: THE 12 OVERARCHING MESSAGES ---\n")
            for i, msg in enumerate(messages_list):
                out_file.write(f"{i+1}. {msg}\n")
            out_file.write("\n")
        except json.JSONDecodeError:
            print("Error parsing JSON from GPT-5. Exiting thread.")
            return

        # --- STEP 4 ---
        print("GPT-5 Step 4: Drafting the 12 LinkedIn posts iteratively...")
        out_file.write("="*50 + "\n--- FINAL LINKEDIN POST DRAFTS ---\n" + "="*50 + "\n\n")
        
        for index, message in enumerate(messages_list):
            print(f"GPT-5 Generating Post {index + 1} of 12...")
            prompt_5 = f"""
            Theme for this post: "{message}"
            
            Generate a LinkedIn post around this theme that delivers practical advice and/or example(s) throughout the post, culminating 
            in a succinct conclusion. INCORPORATE ELEMENTS FROM APPROVED POSTS AND AVOID ELEMENTS FROM REJECTED POSTS.
            
            If this theme mentions an ABM target, strictly follow the ingress strategy from their profile, mention them favorably, and structure the post to naturally encourage them to reach out or accept a meeting with {client_name}.
            
            The example should be rooted in {client_name}'s real or PLAUSIBLY real experiences.
            Ensure you strictly apply a snappy, succinct writing style as we analyzed in the previous step.
            """
            messages.append({"role": "user", "content": prompt_5})
            resp_5 = openai_client.chat.completions.create(model="gpt-5", messages=messages)
            resp_5_text = resp_5.choices[0].message.content
            messages.append({"role": "assistant", "content": resp_5_text})
            
            out_file.write(f"POST {index + 1} THEME: {message}\n")
            out_file.write("-" * 25 + "\n")
            out_file.write(f"{resp_5_text}\n\n")
            out_file.write("*" * 50 + "\n\n")

def run_claude_posts_workflow(client_name, output_filepath, base_text, acc_text, blk_text, abm_text):
    print("Initializing Claude Chat Session...")
    messages = []
    sys_prompt = "You are a professional LinkedIn ghostwriter."
    
    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"RUAN MEI LINKEDIN POSTS (CLAUDE): {client_name.upper()}\n")
        out_file.write("="*50 + "\n\n")

        # --- STEP 1 ---
        print("\nClaude Step 1: Feeding context...")
        prompt_1 = f"""
        I am attaching several files. Among these are interview transcripts between our organization Virio (an organization that writes LinkedIn posts for startup founders and executives)
        and a startup founder/executive named {client_name}, as well as a PDF export of their LinkedIn profile. 
        Comprehend the LinkedIn profile and these interview transcripts thoroughly before we proceed.
        """
        messages.append({"role": "user", "content": prompt_1 + f"\n\nFILES:\n{base_text}"})
        resp_1 = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
        resp_1_text = resp_1.content[0].text
        messages.append({"role": "assistant", "content": resp_1_text})
        out_file.write(f"--- STEP 1: CONTEXT INGESTION ---\n{resp_1_text}\n\n")
        print("Waiting 60 seconds to respect Anthropic rate limits...")
        time.sleep(60)

        # --- STEP 1.1 ---
        if acc_text:
            print("Claude Step 1.1: Analyzing Accepted Posts...")
            prompt_1a = """
            I am now attaching files containing LinkedIn posts that this client has previously APPROVED and loved.
            Thoroughly analyze these approved posts. What specific tone, formatting choices, sentence structures, and overarching themes does the client prefer?
            """
            messages.append({"role": "user", "content": prompt_1a + f"\n\nAPPROVED POSTS:\n{acc_text}"})
            resp_1a = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
            resp_1a_text = resp_1a.content[0].text
            messages.append({"role": "assistant", "content": resp_1a_text})
            out_file.write(f"--- STEP 1.1: APPROVED POSTS ANALYSIS ---\n{resp_1a_text}\n\n")
            print("Waiting 60 seconds to respect Anthropic rate limits...")
            time.sleep(60)

        # --- STEP 1.2 ---
        if blk_text:
            print("Claude Step 1.2: Analyzing Rejected Posts...")
            prompt_1b = """
            I am now attaching files containing LinkedIn posts that this client has REJECTED.
            Thoroughly analyze these rejected posts. Contrast them with the approved ones. What specific content inaccuracies, tones, cliches, or formatting choices must we strictly AVOID moving forward?
            """
            messages.append({"role": "user", "content": prompt_1b + f"\n\nREJECTED POSTS:\n{blk_text}"})
            resp_1b = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
            resp_1b_text = resp_1b.content[0].text
            messages.append({"role": "assistant", "content": resp_1b_text})
            out_file.write(f"--- STEP 1.2: REJECTED POSTS ANALYSIS (AVOID) ---\n{resp_1b_text}\n\n")
            print("Waiting 60 seconds to respect Anthropic rate limits...")
            time.sleep(60)

        # --- STEP 2 ---
        print("Claude Step 2: Analyzing ICP...")
        prompt_2 = f"First, what is {client_name}'s ideal customer profile? What is his/her product? How would his/her product and philosophies resonate with his/her ideal customer?"
        messages.append({"role": "user", "content": prompt_2})
        resp_2 = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
        resp_2_text = resp_2.content[0].text
        messages.append({"role": "assistant", "content": resp_2_text})
        out_file.write(f"--- STEP 2: ICP & PRODUCT ANALYSIS ---\n{resp_2_text}\n\n")
        print("Waiting 60 seconds to respect Anthropic rate limits...")
        time.sleep(60)
        
        # --- STEP 2.5 ---
        abm_instruction = ""
        if abm_text:
            print("Claude Step 2.5: Ingesting ABM Profiles...")
            prompt_2_5 = f"""
            I am attaching Account-Based Marketing (ABM) target profiles. Please review these targets, their predicted pain points, and recommended ingress strategies.
            \n\nABM PROFILES:\n{abm_text}
            """
            messages.append({"role": "user", "content": prompt_2_5})
            resp_2_5 = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
            resp_2_5_text = resp_2_5.content[0].text
            messages.append({"role": "assistant", "content": resp_2_5_text})
            out_file.write(f"--- STEP 2.5: ABM PROFILES INGESTION ---\n{resp_2_5_text}\n\n")
            abm_instruction = "CRITICAL: Exactly 3 of these 12 messages MUST be Account-Based Marketing (ABM) posts dedicated to the specific targets provided in the ABM profiles. These ABM messages should favorably mention the target, address their specific pain points, and subtly position {client_name} as the solution to encourage a meeting."
            print("Waiting 60 seconds to respect Anthropic rate limits...")
            time.sleep(60)

        # --- STEP 3 ---
        print("Claude Step 3: Developing 12 overarching messages...")
        prompt_3 = f"""
        We need 12 posts that range from BOFU to MOFU to TOFU. 
        Begin with drafting 12 compelling overarching messages that you would like our posts to deliver. 
        Let's leverage a variety of {client_name}'s practices and philosophies to appeal to {client_name}'s ICP.
        {abm_instruction}
        CRITICAL INSTRUCTION: Output your response EXCLUSIVELY as a valid JSON array of 12 strings. 
        """
        messages.append({"role": "user", "content": prompt_3})
        resp_3 = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
        resp_3_text = resp_3.content[0].text
        messages.append({"role": "assistant", "content": resp_3_text})
        
        try:
            clean_json = resp_3_text.replace("```json", "").replace("```", "").strip()
            if "[" in clean_json and "]" in clean_json:
                clean_json = clean_json[clean_json.find("["):clean_json.rfind("]")+1]
            messages_list = json.loads(clean_json)
            out_file.write("--- STEP 3: THE 12 OVERARCHING MESSAGES ---\n")
            for i, msg in enumerate(messages_list):
                out_file.write(f"{i+1}. {msg}\n")
            out_file.write("\n")
        except json.JSONDecodeError:
            print("Error parsing JSON from Claude. Exiting thread.")
            return
        
        print("Waiting 60 seconds to respect Anthropic rate limits...")
        time.sleep(60)

        # --- STEP 4 ---
        print("Claude Step 4: Drafting the 12 LinkedIn posts iteratively...")
        out_file.write("="*50 + "\n--- FINAL LINKEDIN POST DRAFTS ---\n" + "="*50 + "\n\n")
        
        for index, message in enumerate(messages_list):
            print(f"Claude Generating Post {index + 1} of 12...")
            prompt_5 = f"""
            Theme for this post: "{message}"
            
            Generate a LinkedIn post around this theme that delivers practical advice and/or example(s) throughout the post, culminating 
            in a succinct conclusion. INCORPORATE ELEMENTS FROM APPROVED POSTS AND AVOID ELEMENTS FROM REJECTED POSTS.
            
            If this theme mentions an ABM target, strictly follow the ingress strategy from their profile, mention them favorably, and structure the post to naturally encourage them to reach out or accept a meeting with {client_name}.
            
            The example should be rooted in {client_name}'s real or PLAUSIBLY real experiences.
            Ensure you strictly apply a snappy, succinct writing style as we analyzed in the previous step.
            """
            messages.append({"role": "user", "content": prompt_5})
            resp_5 = anthropic_client.messages.create(model="claude-opus-4-6", max_tokens=4096, system=sys_prompt, messages=messages)
            resp_5_text = resp_5.content[0].text
            messages.append({"role": "assistant", "content": resp_5_text})
            
            out_file.write(f"POST {index + 1} THEME: {message}\n")
            out_file.write("-" * 25 + "\n")
            out_file.write(f"{resp_5_text}\n\n")
            out_file.write("*" * 50 + "\n\n")
            
            # Additional sleep after each post to avoid Anthropic rate limits
            print("Waiting 20 seconds to respect Anthropic rate limits...")
            time.sleep(20)

def generate_iterative_linkedin_posts(client_name, company_keyword, model_choice="All (Ensemble)"):
    directory_path = f"./client_data/{company_keyword}"
    output_path = os.path.join(directory_path, "output")
    accepted_path = os.path.join(directory_path, "accepted")
    blocked_path = os.path.join(directory_path, "rejected")
    abm_path = os.path.join(directory_path, "abm_profiles")
    
    google_output_filename = f"{company_keyword}_gemini_posts.md"
    google_output_filepath = os.path.join(output_path, google_output_filename)
    gpt_output_filename = f"{company_keyword}_gpt_posts.md"
    gpt_output_filepath = os.path.join(output_path, gpt_output_filename)
    claude_output_filename = f"{company_keyword}_claude_posts.md"
    claude_output_filepath = os.path.join(output_path, claude_output_filename)
    
    final_output_filename = f"{company_keyword}_posts.md"
    final_output_filepath = os.path.join(output_path, final_output_filename)

    print(f"Scanning directories for {client_name}...")
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return
    
    # 1. Load context based on model choice
    base_files, accepted_files, blocked_files, abm_files = [], [], [], []
    local_context, acc_posts, blk_posts, abm_posts = "", "", "", ""
    
    if model_choice in ["All (Ensemble)", "Gemini 3.1 Pro"]:
        base_files = upload_and_wait(directory_path, google_client, skip_files=[f for f in os.listdir(directory_path) if f.startswith(company_keyword)])
        accepted_files = upload_and_wait(accepted_path, google_client, skip_files=[])
        blocked_files = upload_and_wait(blocked_path, google_client, skip_files=[])
        abm_files = upload_and_wait(abm_path, google_client, skip_files=[])
        
    if model_choice in ["All (Ensemble)", "GPT-5", "Claude Opus 4.6"]:
        local_context = get_local_context(directory_path, skip_files=[f for f in os.listdir(directory_path) if f.startswith(company_keyword)])
        acc_posts = "\n--- APPROVED POSTS ---\n" + get_local_context(accepted_path, skip_files=[]) if os.path.exists(accepted_path) else ""
        blk_posts = "\n--- REJECTED POSTS ---\n" + get_local_context(blocked_path, skip_files=[]) if os.path.exists(blocked_path) else ""
        abm_posts = "\n--- ABM TARGET PROFILES ---\n" + get_local_context(abm_path, skip_files=[]) if os.path.exists(abm_path) else ""

    # 2. Execution Routing
    if model_choice == "Gemini 3.1 Pro":
        print(f"Running Solo Post Generation with Gemini for {client_name}...")
        run_gemini_posts_workflow(client_name, google_output_filepath, base_files, accepted_files, blocked_files, abm_files)
        # Duplicate to final path so GUI loads it easily
        import shutil
        shutil.copy(google_output_filepath, final_output_filepath)
        
    elif model_choice == "GPT-5":
        print(f"Running Solo Post Generation with GPT-5 for {client_name}...")
        run_gpt5_posts_workflow(client_name, gpt_output_filepath, local_context, acc_posts, blk_posts, abm_posts)
        import shutil
        shutil.copy(gpt_output_filepath, final_output_filepath)
        
    elif model_choice == "Claude Opus 4.6":
        print(f"Running Solo Post Generation with Claude for {client_name}...")
        run_claude_posts_workflow(client_name, claude_output_filepath, local_context, acc_posts, blk_posts, abm_posts)
        import shutil
        shutil.copy(claude_output_filepath, final_output_filepath)
        
    else:
        # Full Ensemble Mode WITH SYNTHESIS
        print(f"Triggering Parallel Posts Ensemble for {client_name}...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_gemini = executor.submit(run_gemini_posts_workflow, client_name, google_output_filepath, base_files, accepted_files, blocked_files, abm_files)
            f_gpt = executor.submit(run_gpt5_posts_workflow, client_name, gpt_output_filepath, local_context, acc_posts, blk_posts, abm_posts)
            f_claude = executor.submit(run_claude_posts_workflow, client_name, claude_output_filepath, local_context, acc_posts, blk_posts, abm_posts)
            
            # Wait for all generators to finish before proceeding to synthesis
            f_gemini.result()
            f_gpt.result()
            f_claude.result()
            
        print("Ensemble generation complete. Synthesizing posts using Cyrene...")
        
        try:
            # 1. Read the drafted outputs from the individual files
            with open(google_output_filepath, "r", encoding="utf-8") as f:
                gemini_draft = f.read()
            with open(gpt_output_filepath, "r", encoding="utf-8") as f:
                gpt_draft = f.read()
            with open(claude_output_filepath, "r", encoding="utf-8") as f:
                claude_draft = f.read()
            
            # Package them into the dict format Cyrene expects
            raw_drafts = {
                "Gemini 3.1 Pro": gemini_draft,
                "GPT-5": gpt_draft,
                "Claude Opus 4.6": claude_draft
            }
            
            # 2. Use Cyrene to synthesize drafts
            cyrene = Cyrene()
            synthesis = cyrene.synthesize_post(raw_drafts=raw_drafts)

            # 3. Write final synthesized output to the final output file
            with open(final_output_filepath, "w", encoding="utf-8") as out_file:
                out_file.write(f"# SYNTHESIZED MASTER POSTS: {client_name.upper()}\n\n")
                out_file.write("## Synthesis Strategy\n")
                out_file.write(f"{synthesis.get('strategy', 'N/A')}\n\n")
                out_file.write("## FINAL SYNTHESIZED POST(S)\n")
                out_file.write(f"{synthesis.get('synthesized_draft', 'N/A')}\n")
                
            print(f"Synthesis successful! Saved to {final_output_filepath}")
            
        except Exception as e:
            print(f"Synthesis failed with error: {e}")
            # Fallback to the original placeholder if something goes wrong during synthesis
            with open(final_output_filepath, "w", encoding="utf-8") as out_file:
                out_file.write(f"# ENSEMBLE GENERATION COMPLETE: {client_name.upper()}\n\n")
                out_file.write(f"Synthesis failed due to error: {e}\n\n")
                out_file.write("Please review the individual model outputs in this directory:\n")
                out_file.write(f"- Gemini 3.1 Pro: `{google_output_filename}`\n")
                out_file.write(f"- GPT-5: `{gpt_output_filename}`\n")
                out_file.write(f"- Claude Opus 4.6: `{claude_output_filename}`\n")

    print(f"\nProcess complete! Please check the output at: {final_output_filepath}")