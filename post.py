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

def generate_iterative_linkedin_posts(client_name, company_keyword):
    directory_path = f"./client_data/{company_keyword}"
    accepted_path = os.path.join(directory_path, "accepted")
    blocked_path = os.path.join(directory_path, "blocked")
    
    output_filename = f"{company_keyword}_ruanmei_posts.txt"
    output_filepath = os.path.join(directory_path, output_filename)
    briefing_filename = f"{company_keyword}_ruanmei_briefing.md"

    print(f"Scanning directory: {directory_path}...")
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return
    
    base_files = upload_and_wait(directory_path, client, skip_files=[output_filepath, briefing_filename])
    accepted_files = upload_and_wait(accepted_path, client, skip_files=[])
    blocked_files = upload_and_wait(blocked_path, client, skip_files=[])

    all_uploaded_files = base_files + accepted_files + blocked_files

    print("Files ready. Initializing Chat Session...")

    chat = client.chats.create(
        model="gemini-3.1-pro-preview",
        config=types.GenerateContentConfig(temperature=0.7)
    )

    # Open the text file to start recording the session
    with open(output_filepath, "w", encoding="utf-8") as out_file:
        out_file.write(f"RUAN MEI LINKEDIN POSTS: {client_name.upper()}\n")
        out_file.write("="*50 + "\n\n")

        # --- STEP 1: Context Ingestion ---
        print("\nStep 1: Feeding context...")
        prompt_1 = f"""
        I am attaching several files. Among these are interview transcripts between our organization Virio (an organization that writes LinkedIn posts for startup founders and executives)
        and a startup founder/executive named {client_name}, as well as a PDF export of their LinkedIn profile. 
        Comprehend the LinkedIn profile and these interview transcripts thoroughly before we proceed.
        """
        response_1 = chat.send_message([prompt_1] + base_files)
        out_file.write(f"--- STEP 1: CONTEXT INGESTION ---\n{response_1.text}\n\n")

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
            Thoroughly analyze these rejected posts. Contrast them with the approved ones. What specific content inaccuracies, tones, cliches, or formatting choices must we strictly AVOID moving forward?
            """
            response_1b = chat.send_message([prompt_1b] + blocked_files)
            out_file.write(f"--- STEP 1.2: REJECTED POSTS ANALYSIS (AVOID) ---\n{response_1b.text}\n\n")

        # --- STEP 2: ICP & Product Analysis ---
        print("Step 2: Analyzing ICP...")
        prompt_2 = f"First, what is {client_name}'s ideal customer profile? What is his/her product? How would his/her product and philosophies resonate with his/her ideal customer?"
        response_2 = chat.send_message(prompt_2)
        out_file.write(f"--- STEP 2: ICP & PRODUCT ANALYSIS ---\n{response_2.text}\n\n")

        # --- STEP 3: Overarching Strategy ---
        print("Step 3: Developing 10 overarching messages...")
        prompt_3 = f"""
        We need 10 posts that range from BOFU to MOFU to TOFU. 
        Begin with drafting 10 compelling overarching messages that you would like our posts to deliver. 
        Let's leverage a variety of {client_name}'s practices and philosophies to appeal to {client_name}'s ICP.
        CRITICAL INSTRUCTION: Output your response EXCLUSIVELY as a valid JSON array of 10 strings. 
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
        Before we write the posts, thoroughly read and comprehend the style, tone, and structure 
        of the following perfect example posts:
    
Strategic "Roadblocking" for Culture

“Raise money, then hire as fast as you can.”

That's the standard Silicon Valley advice. If you have the budget, you fill the seats.

Growth is everything, right?

I disagree.

Adding headcount to a process that can't handle more weight doesn't fix any problems… it just makes failures all the more expensive.

When my Head of Marketing joined Runpod, he did exactly what he was supposed to do: he asked for a team. We had the revenue. We had the workload. My co-founder, Pardeep, has a "bias for action" that moves mountains… he was ready to hire immediately.

I said no.

I roadblocked the department for a month. Why?

Because, before we added people, I needed to know three things:

Is the process repeatable? If a leader can't document exactly how they won, a new hire will spend six months guessing.

Is this a "spike" or a "new baseline"? We don't hire for a busy week; we hire for a permanent shift in volume.

What is the "Management Tax"? If a leader spends 80% of their time managing people, their actual output drops. Can the department survive that dip?

Too often, we obsess over the speed of the climb and forget about structural integrity.

As a leader, my job isn't just to step on the gas. It's to know when to tap the brakes.

-> Scaling is an engineering problem, not a spending problem.

We all dream of flying ever closer to the sun. But first, we need to make sure that our wings can survive the heat of our ambitions.

//

Taste as the Ultimate Competitive Advantage

AI is engineered to regress to the mean.

Think about the fundamental architecture of a large language model. It is trained on the sum of human history to predict the most statistically probable next outcome.

In business, "statistically probable" is another word for "market average."

When every one of your competitors has the same supercomputer on their desk churning out infinite, perfectly average work, the rarest commodity on the market isn't execution. It's asymmetry.

In a landscape where the cost of generating output is trending toward zero, the only way to maintain a competitive advantage is to apply "Taste." This isn't just an abstract concept; it is a practical workflow for any technical founder or engineer.

Here is how you apply it:

1. Identify the "Default" Solution

If you ask an AI for a product roadmap, a marketing hook, or a code architecture, identify exactly what it suggests first. That is now your "Baseline of the Average." Nothing more. If you ship that, you are shipping a commodity that anyone else can replicate in 30 seconds.

2. Optimize for Variance, Not Volume

The machine will always beat you on volume. Don't try to out-produce the model. Instead, your job is to intentionally "push" the output away from the statistical center. Ask: “What is the machine NOT statistically likely to suggest here?” That is where the value lives.

3. Lean into "Individualist" Culture

In our content interview, we discussed how the West's individualist culture is actually a structural advantage in the AI era. We are fundamentally wired to stand out and break the mold. In a global market of automated agents, being "loud" and having a deeply individual point of view is a defensive moat.

4. The "Push and Pull" Workflow

The future of work is a relentless game of push and pull. The autonomous machine will constantly try to pull your work into the "good enough" center. Your job is to have the technical and creative taste to push it back out to the fringes.

Success in the next five years won't go to the person who prompts the best. It will go to the person who has the audacity to articulate a vision that the machine could never have predicted.

Don't compete with the machine on volume. Compete on variance.

//

The Case for the "Slow Roll" Launch

The startup playbook demands a spectacle.

A cinematic launch video. A coordinated Product Hunt strike. We treat product launches like Hollywood premieres, obsessing over the red carpet.

In college, I took drama and improv classes to force myself out of my "shy kid" shell. One of the first things you study is Aristotle's framework for theater.

He ranked the elements of a play by importance. The absolute most important? The plot. The structural foundation of the story. The absolute least important? The spectacle. The special effects.

Silicon Valley has forgotten this. We are obsessed with the spectacle.

When we launched RunPod, we skipped the special effects entirely. No manufactured "waitlist theater." No PR agency pulling strings in private Slack channels. We executed what I call a "slow roll."

We launched on a Sunday evening. On Reddit.

Why Reddit? Because developers don't go to Reddit to applaud slick marketing videos. They go there to brutally dissect tools, bypass corporate fluff, and complain about the friction in their current stack. If your product relies on hype, a highly technical forum will tear it apart in minutes.

But if you actually solve a structural problem, it is the highest-value proving ground on earth.

Instead of a polished landing page, we just handed a piece of raw compute infrastructure directly to the people who desperately needed it. And we completely ignored standard launch metrics. We didn't care about impressions, upvotes, or newsletter signups.

We tracked exactly one practical metric: Time to First Deployment. If an engineer clicked our Reddit link, how many minutes did it take for them to actually spin up a GPU and start running code? We optimized the entire product to make that friction point disappear.

When you focus entirely on the core utility instead of the launch spectacle, a shift happens. Engineers don't just applaud your launch and leave. They adopt the tool. They integrate it into their daily workflows. They become the unshakeable foundation of your momentum.

A hype launch gives you a viral spike on Monday and a 90% churn rate by Friday. A slow roll gives you a community.

Stop obsessing over the blast radius of your launch day.

Build for the quiet Tuesday after the smoke clears.

//

The Introvert's Journey to CEO

At 17, my comfort zone was a computer screen. I was a profoundly shy kid, far more comfortable with the raw logic of complex strategy games than I was with actual human beings.

But raw logic doesn't scale a company. People do.

I realized early on that my analytical engine was fine, but my "human interface" was bottlenecked by introversion. If I wanted to build anything meaningful, I had to deliberately engineer my way out of it.

So, I forced myself into the most uncomfortable room imaginable: a college acting class.

It sounds counterintuitive, but theater gave me the ultimate framework for executive leadership.

It's rooted in a foundational practice called the Meisner Technique.

The core instruction of Meisner is surprisingly simple: Take all of your attention off yourself, and place it entirely on your scene partner.

When you are obsessively observing how another person breathes, speaks, and reacts, a fascinating thing happens: you literally do not have the cognitive bandwidth left to be shy.

You stop overthinking your own awkwardness. You bypass your internal social friction by simply responding to their reality.

Theater didn't magically cure my introversion. I didn't suddenly become a relentless extrovert.

But it taught me that "presence" and "empathy" aren't just mysterious soft skills that you are either born with or you aren't. They are trainable frameworks. They are a deliberate focus of attention.

Leadership isn't a genetic trait reserved for the loudest person in the room.

You can possess the deep, systemic intelligence of a quiet engineer and deliberately equip the active listening of a stage actor.

You just have to be willing to install the upgrade yourself.

//

The Power of the "Push and Pull" Partnership

If you and your co-founder agree on everything, one of you is redundant.

The startup world romanticizes "perfectly aligned" co-founders. We're told that partners should share a single brain and move in absolute lockstep. Total consensus is praised as a green light.

I see it as a massive red flag.

My co-founder, Pardeep, is a force of nature. He has a relentless "bias for action." If he sees an opportunity, his instinct is to run straight at it. My default state is the exact opposite: "intentional thoughtfulness." I want to stress-test the operational debt before we make a move.

To an outsider, this looks like a recipe for gridlock. But in practice, it is our greatest structural advantage.

Let me give you a real example from RunPod.

A while ago, we saw a massive spike in demand from European developers. Pardeep found a Tier-1 facility in Frankfurt with immediate rack space. His instinct was immediate: “If we don't sign this lease by Friday, a competitor will. We need to move now.”

My instinct wasn't to say yes or no, but to pull back and ask the structural questions: “If we move that fast, do we have the localized on-call rotation to support 3 AM outages in Frankfurt time? Have we stress-tested the backhaul latency? If we sign this Friday, are we stacking the weight of a new continent on a DevOps team that is already at full capacity?”

If Pardeep ran the company alone, we might have launched immediately and crashed under the technical weight. If I ran the company alone, we would have over-analyzed the latency and missed the market window entirely.

Instead, we used our opposing instincts as a decision-making engine. We didn't compromise; we synthesized.

We signed the lease that Friday to secure the space, but we deliberately delayed the public launch by 14 days to bake in a new automated failover protocol that I insisted on. Pardeep got the speed. I got the integrity. The result was a high-velocity launch that actually stayed online.

A resilient company isn't built on absolute agreement. It is built on the deliberate, healthy friction between speed and due diligence.

Don't look for a co-founder who acts as your echo chamber.

Look for your counterweight.

//

Infrastructure as Invisible "Scaffolding"

When a master architect designs a skyscraper, they obsess over the steel, the glass, and the silhouette it will cast against the skyline.

They do not obsess over the scaffolding used to build it.

Scaffolding has exactly one job: to hold the weight of the builder without drawing attention to itself. The moment a builder feels the planks bowing under their feet, their focus violently shifts from creating their masterpiece to ensuring their own survival.

In the AI ecosystem, developers are the architects. Compute is the scaffolding.

Yet, the current landscape of cloud computing forces you to stare at the scaffolding all day. You spend half your cognitive bandwidth fighting complex DevOps configurations, navigating arbitrary quota limits, and trying to decode labyrinthine billing structures.

You are spending your time trying not to fall off the planks instead of actually building the tower.

-> If you have to think about your infrastructure, your cloud provider has failed in their primary job.

That is the exact premise we built Runpod on. We don't want to be the center of your attention. We don't want to trap you in a walled garden of proprietary tools just to inflate our ecosystem.

Our engineering philosophy is "invisible infrastructure."

We provide raw, unthrottled, highly accessible compute that stays entirely out of your way. We handle the load-bearing reality of GPU orchestration so you can get back to the only thing that actually matters: your product.

Great infrastructure shouldn't give you more tools to manage. It should give you your focus back.

Stop managing the scaffolding. Start building the skyline.

//

Solving Our Own "Hot Garbage" Problem

Startup lore says great companies begin with a visionary whiteboard session. You study the market trends. You run the focus groups. You find the "white space."

I completely disagree. The most enduring products aren't born from market research.

They are born from sheer, unadulterated frustration.

Imagine trying to cook a Michelin-star meal, but every time you need to turn on the stove, you have to navigate a labyrinth of billing configurations and file a request form just to get gas. You wouldn't just keep cooking. You'd tear the kitchen down and build a new one.

That is exactly how Runpod started.

Pardeep and I weren't sitting around dreaming of building a cloud infrastructure company. We were just two developers trying to build AI applications. But the existing GPU clouds were, frankly, hot garbage.

We spent 80% of our time fighting the infrastructure—wrestling with obscure quotas, broken dependencies, and clunky DevOps—and only 20% actually writing code.

We were our own most frustrated customers.

So, we stopped trying to build apps on top of a broken foundation. We pivoted and built the infrastructure we desperately wished existed. Runpod exists for one simple reason: we wanted a platform that didn't make us hate our lives on a Tuesday afternoon.

You don't need a focus group to tell you a tool is broken. You just need the scars from trying to use it.

The best products aren't built by visionaries looking down at a market. They are built by frustrated users looking up from the trenches.

//

Human-to-Human Scaling in a $120M Business

The startup playbook has a very specific timeline for listening to your users.

At $1M ARR, you are in the trenches.

At $10M, you hire a VP to be in the trenches.

At $100M, you read a summarized quarterly report about the trenches from a sanitized dashboard.

“Do things that don't scale,” they say. But only until you scale.

I fundamentally disagree. In the AI ecosystem, retreating to the executive dashboard is an irresponsible ticket to obsolescence.

Runpod recently crossed a massive revenue milestone and supports over 500,000 developers. By traditional Silicon Valley standards, my co-founder and I should be entirely insulated by now. We should be looking at moving averages, not reading bug reports.

But my most valuable strategic insights don't come from board meetings. They come from a midnight Discord chat with a frustrated engineer trying to orchestrate a cluster of GPUs.

In a landscape where the underlying technology completely resets every six months, you cannot afford a delayed signal. By the time a market shift filters up through four layers of management into a PowerPoint presentation, the frontier has already moved.

You have to maintain that raw, unfiltered "founder-to-founder" level of communication. You need to feel the heat the second the friction happens, not a month later in a quarterly review.

Scale your infrastructure. Scale your headcount.

Never scale the distance between the builder and the user.

//

WHAT IS YOUR MOAT

Every pitch meeting inevitably hits the exact same question:

“What is your moat?”

The Silicon Valley playbook demands that you build high walls. Lock users into your proprietary ecosystem. Hoard your data. Create a fortress so impenetrable that your competitors starve on the outside.

I disagree. A fortress is just a tomb with a heavy door.

My academic background is in physical chemistry. In thermodynamics, there is a very simple rule: if you want to guarantee the death of a system, you close it off.

A "closed system" cannot exchange energy with the outside world. Without that external friction and flow, it inevitably succumbs to entropy. The internal energy flatlines. The system dies.

When you build a traditional tech "moat," you are purposefully engineering a closed system. You might successfully trap your users for a while, but you are also trapping your own innovation. You stop reacting to the outside environment because you falsely believe your walls will protect you.

We see this constantly in legacy cloud infrastructure. They build massive, convoluted walled gardens to prevent developers from leaving.

But in the AI era, the landscape is entirely seismic. Proprietary ecosystems are being outpaced and dismantled by open-source, interoperable, high-velocity developer communities in real-time.

If you spend all your time digging a moat, you won't notice that the river has already changed course.

The era of static defense is over. In a market that fundamentally resets every six months, walls will not save you. Adaptability will.

-> Your moat isn't your masonry. It's your momentum.

//

The "Privatization of Education" in the AI Era

“Get the degree. Secure the future.”

That was the promise.

But in the AI era, a four-year degree is nothing but a static map for an already fully-explored, well-charted landscape.

It does nothing to prepare us for braving the ever-shifting frontiers of science and industry.

I spent years in academia earning my PhD. I value the rigor. But we have to be honest: traditional institutions cannot keep pace with the current speed of innovation.

By the time a syllabus is approved by a board, the breakthroughs it covers are already relics.

The "Privatization of Education" is no longer a theory. It's our current reality.

This education isn't happening in lecture halls. It's happening in our phones and laptops: in private Discord servers, open-source repos, and GPU clusters. It's happening in the "trenches" where the compute lives.

As professionals learning as we explore, we are like Icarus, building our wings while already in flight. But in this race, the danger isn't just flying too high.

It's flying too slow.

If your knowledge is static, the wax doesn't just melt; the sun moves away from you.

The only asset that doesn't depreciate?

A relentless thirst for knowledge.

Success in AI isn't about what you've already memorized. It's about your ability to learn how to learn. It's the ability to ingest new information at scale, discard what is no longer true, and reinvent your workflow on the fly.

In a world of autonomous agents and shifting models, "taste" and "thirst" are the only things AI can't replace.

Don't just chase the "what." Master the "how."

The frontier has moved from the classroom to the cluster. 
        
        Break down exactly what makes the writing and line break choices in these examples succinct, snappy, and effective. 
        Acknowledge how they debunk widespread ideas through practical advice.
        """
        response_4 = chat.send_message(prompt_4)
        out_file.write(f"--- STEP 4: EXAMPLE ANALYSIS ---\n{response_4.text}\n\n")

        # --- STEP 5: Iterative Generation ---
        print("Step 5: Drafting the 10 LinkedIn posts iteratively...")
        out_file.write("="*50 + "\n--- FINAL LINKEDIN POST DRAFTS ---\n" + "="*50 + "\n\n")
        
        for index, message in enumerate(messages_list):
            print(f"Generating Post {index + 1} of 10...")
            prompt_5 = f"""
            Theme for this post: "{message}"
            
            Generate a LinkedIn post around this theme that begins with a commonplace occurrence 
            or a widespread idea that is debunked through practical advice and/or example(s) throughout the post, culminating 
            in a succinct conclusion. 
            
            The example should be rooted in {client_name}'s real or PLAUSIBLY real experiences.
            Ensure you strictly apply a snappy, succinct writing style as we analyzed in the previous step.
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