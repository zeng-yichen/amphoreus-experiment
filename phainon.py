"""
Phainon — Post generation dispatcher.

Routes to Stelle (jacquard-style agentic ghostwriter) for all post generation.
Also provides utility functions used by the GUI (context loading, ABM names).
"""

import os

import vortex as P


def get_local_context(directory: str, skip_files: list) -> str:
    """Read .txt and .md files from a directory into a single string.

    Used by amphoreus.py to load client context for Cyrene rewrites.
    """
    context_text = ""
    if not os.path.exists(directory):
        return context_text
    for filename in os.listdir(directory):
        if filename in skip_files:
            continue
        filepath = os.path.join(directory, filename)
        if filename.lower().endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                context_text += f"\n--- DOCUMENT: {filename} ---\n{f.read()}\n"
    return context_text


def generate_posts(client_name: str, company_keyword: str) -> str:
    """Generate LinkedIn posts via Stelle agentic workflow.

    Returns the path to the output markdown file.
    """
    from stelle import generate_one_shot

    P.ensure_dirs(company_keyword)
    output_dir = P.post_dir(company_keyword)
    output_filepath = str(output_dir / f"{company_keyword}_posts.md")

    print(f"[Phainon] Starting post generation for {client_name}...")
    result_path = generate_one_shot(client_name, company_keyword, output_filepath)
    print(f"\n[Phainon] Generation complete! Output at: {result_path}")
    return result_path
