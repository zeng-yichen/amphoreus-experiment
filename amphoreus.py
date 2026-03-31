import os
import platform
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import shutil
import threading
import json
import sys
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from PIL import Image, ImageTk

from backend.src.db import vortex as P
from backend.src.agents.stelle import generate_one_shot
from backend.src.agents.aglaea import generate_briefing
from backend.src.agents.anaxa import Anaxa
from backend.src.agents.cerydra import Cerydra
from backend.src.agents.cyrene import Cyrene, MAX_BATCH_POSTS
from backend.src.agents.hysilens import Hysilens
from backend.src.agents.hyacinthia import Hyacinthia
from backend.src.agents import screwllum

class PrintLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
    def flush(self):
        pass

def _read_local_context(directory: str, skip_files: list) -> str:
    text = ""
    if not os.path.exists(directory):
        return text
    for fn in os.listdir(directory):
        if fn in skip_files:
            continue
        if fn.lower().endswith((".txt", ".md")):
            with open(os.path.join(directory, fn), "r", encoding="utf-8") as f:
                text += f"\n--- DOCUMENT: {fn} ---\n{f.read()}\n"
    return text


class AmphoreusExperiment:
    def __init__(self, root):
        self.root = root
        self.root.title("The Amphoreus Experiment")
        self.root.geometry("1500x950") 
        self.root.configure(padx=10, pady=10, bg="white")

        # --- STATE ---
        self.anaxa = Anaxa()
        self.cerydra = Cerydra()
        self.hysilens = Hysilens()

        # --- FONTS ---
        self.UI_FONT = ("Arial", 11, "bold")
        self.HOVER_FONT = ("Arial", 12, "bold") 
        self.HEADER_FONT = ("Arial", 12, "bold")

        # --- TREEVIEW STYLING (To bold the Data Explorer tree) ---
        style = ttk.Style()
        style.configure("Treeview", font=("Arial", 10, "bold"))
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"))

        self.chibi_images = []
        self.image_refs = []
        static_dir = os.path.join(os.path.dirname(__file__), "backend", "static", "images")
        
        if os.path.exists(static_dir):
            for file in os.listdir(static_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    if file.lower() not in ["aglaea.jpg", "phainon.jpg"]:
                        self.chibi_images.append(os.path.join(static_dir, file))

        self.aglaea_img = self._load_specific_image(os.path.join(static_dir, "aglaea.jpg"), size=(30, 30))
        self.phainon_img = self._load_specific_image(os.path.join(static_dir, "phainon.jpg"), size=(30, 30))

        # --- LAYOUT ---
        self.left_panel = tk.Frame(root, width=350, bg="white")
        self.left_panel.pack(side="left", fill="y", expand=False, padx=(0, 10))
        self.left_panel.pack_propagate(False) 
        
        self.right_panel = tk.Frame(root, width=380, bg="white")
        self.right_panel.pack(side="right", fill="y", expand=False, padx=(10, 0))
        self.right_panel.pack_propagate(False)
        
        self.center_panel = tk.Frame(root, bg="white")
        self.center_panel.pack(side="left", fill="both", expand=True)

        # ==========================================
        #               LEFT PANEL
        # ==========================================
        frame_details = tk.LabelFrame(self.left_panel, text="1. Client Details", padx=5, pady=5, bg="white", relief="flat", bd=0, font=self.HEADER_FONT)
        frame_details.pack(fill="x", pady=2)
        frame_details.columnconfigure(1, weight=1)

        tk.Label(frame_details, text="Client Name:", bg="white", font=self.UI_FONT).grid(row=0, column=0, sticky="w")
        self.client_name_var = tk.StringVar()
        tk.Entry(frame_details, textvariable=self.client_name_var, relief="solid", borderwidth=1, highlightthickness=2, highlightbackground="white", highlightcolor="black", font=self.UI_FONT).grid(row=0, column=1, padx=(5, 0), pady=2, sticky="ew")

        tk.Label(frame_details, text="Company Keyword:", bg="white", font=self.UI_FONT).grid(row=1, column=0, sticky="w")
        self.company_var = tk.StringVar()
        tk.Entry(frame_details, textvariable=self.company_var, relief="solid", borderwidth=1, highlightthickness=2, highlightbackground="white", highlightcolor="black", font=self.UI_FONT).grid(row=1, column=1, padx=(5, 0), pady=2, sticky="ew")

        frame_files = tk.LabelFrame(self.left_panel, text="2. Upload Documents", padx=5, pady=5, bg="white", relief="flat", bd=0, font=self.HEADER_FONT)
        frame_files.pack(fill="x", pady=2)
        
        tk.Label(frame_files, text="Make sure Company Keyword is filled out first.", fg="gray", bg="white", font=("Arial", 9, "bold")).pack(pady=(0, 4))
        frame_files.pack_propagate(True)
        
        self._create_clickable_label(frame_files, "Add Base Files (Transcripts/Profile)", lambda: self.show_upload_dialog("base")).pack(fill="x", pady=8)
        self._create_clickable_label(frame_files, "Add Accepted Posts (Optional)", lambda: self.show_upload_dialog("accepted")).pack(fill="x", pady=8)
        self._create_clickable_label(frame_files, "Add Feedback / Revisions (Optional)", self.show_revision_dialog).pack(fill="x", pady=8)
        self._create_clickable_label(frame_files, "Manage Past Posts (Dedup)", self.open_past_posts_dialog).pack(fill="x", pady=8)

        frame_gen = tk.LabelFrame(self.left_panel, text="3. Generation", padx=5, pady=5, bg="white", relief="flat", bd=0, font=self.HEADER_FONT)
        frame_gen.pack(fill="x", pady=2)

        self.action_var = tk.StringVar(value="brief")

        rb_frame1 = tk.Frame(frame_gen, bg="white")
        rb_frame1.pack(fill="x", anchor="w")
        tk.Radiobutton(rb_frame1, text="Generate Briefing?", variable=self.action_var, value="brief", bg="white", activebackground="white", highlightbackground="white", font=self.UI_FONT).pack(side="left")
        if self.aglaea_img:
            tk.Label(rb_frame1, image=self.aglaea_img, bg="white").pack(side="left", padx=(5, 0))

        rb_frame2 = tk.Frame(frame_gen, bg="white")
        rb_frame2.pack(fill="x", anchor="w")
        tk.Radiobutton(rb_frame2, text="Generate Posts?", variable=self.action_var, value="post", bg="white", activebackground="white", highlightbackground="white", font=self.UI_FONT).pack(side="left")
        if self.phainon_img:
            tk.Label(rb_frame2, image=self.phainon_img, bg="white").pack(side="left", padx=(5, 0))

        self.run_btn = self._create_clickable_label(frame_gen, "🚀 RUN TASK", self.run_generation)
        self.run_btn.pack(fill="x", pady=8)
        
        frame_rewrite = tk.LabelFrame(self.left_panel, text="4. Stylistic Rewrite with Cyrene", padx=5, pady=5, bg="white", relief="flat", bd=0, font=self.HEADER_FONT)
        frame_rewrite.pack(fill="x", pady=2)
        
        self._create_clickable_label(
            frame_rewrite,
            "Rewrite single post…",
            self.open_single_post_rewrite_dialog,
        ).pack(fill="x", pady=(2, 6))

        self._create_clickable_label(
            frame_rewrite,
            f"Rewrite multiple posts (paste, up to {MAX_BATCH_POSTS})…",
            self.open_batch_post_rewrite_dialog,
        ).pack(fill="x", pady=(0, 6))
        frame_screwllum = tk.LabelFrame(self.left_panel, text="5. Content Strategy Generation", padx=5, pady=5, bg="white", relief="flat", bd=0, font=self.HEADER_FONT)
        frame_screwllum.pack(fill="x", pady=2)

        self._create_clickable_label(
            frame_screwllum,
            "🧠 Generate Content Strategy",
            self.open_screwllum_dialog,
        ).pack(fill="x", pady=(4, 2))

        tk.Label(
            frame_screwllum,
            text="Uses Company Keyword + memory/ transcripts",
            fg="gray", bg="white", font=("Arial", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        frame_style = tk.LabelFrame(self.left_panel, text="6. Client Tone Instructions", padx=5, pady=5, bg="white", relief="flat", bd=0, font=self.HEADER_FONT)
        frame_style.pack(fill="x", pady=2)

        self._create_clickable_label(
            frame_style,
            "✏️ Edit Content Strategy",
            self.open_content_strategy_dialog
        ).pack(fill="x", pady=4)

        # DEPRECATED: Prompt Graph Tuning UI removed
        # frame_gradient and related buttons have been removed

        tk.Label(self.left_panel, text="Console Output:", bg="white", font=self.UI_FONT).pack(anchor="w", pady=(6, 2))
        self.console = scrolledtext.ScrolledText(self.left_panel, width=40, height=5, bg="#1e1e1e", fg="#00ff00", font=("Courier", 10, "bold"), borderwidth=1, relief="solid", highlightthickness=0)
        self.console.pack(fill="both", expand=True)
        self.console.vbar.configure(troughcolor="white", bg="white", activebackground="#e0e0e0", borderwidth=0)

        sys.stdout = PrintLogger(self.console)

        # ==========================================
        #               CENTER PANEL 
        # ==========================================
        header_frame = tk.Frame(self.center_panel, bg="white")
        header_frame.pack(fill="x", pady=(0, 10))
        
        self._place_chibi(header_frame, side="left", size=(50, 50))
        tk.Label(header_frame, text="Document Viewer", font=("Arial", 16, "bold"), bg="white").pack(side="left", padx=(5, 0))
        
        self._create_clickable_label(header_frame, "🔄 Manually Load Output", self.load_output_to_viewer).pack(side="right", padx=10)

        # --- SEARCH BAR (Using Interactive Text Labels instead of Buttons) ---
        self.doc_search_var = tk.StringVar()
        
        self._create_clickable_label(header_frame, "✖ Clear", self.clear_doc_search).pack(side="right", padx=(5, 10))
        self._create_clickable_label(header_frame, "🔍 Find", self.search_doc_viewer).pack(side="right", padx=(5, 5))
        
        self.doc_search_entry = tk.Entry(header_frame, textvariable=self.doc_search_var, font=self.UI_FONT, width=20, relief="solid", borderwidth=1, highlightthickness=1)
        self.doc_search_entry.pack(side="right")
        self.doc_search_entry.bind("<Return>", lambda event: self.search_doc_viewer())
        
        # ---------------------------------------------------------------------

        # Intentionally unbolded per request: "aside from text in the Document Viewer..."
        self.output_viewer = scrolledtext.ScrolledText(self.center_panel, bg="#ffffff", fg="#333333", font=("Helvetica", 11), wrap="word", relief="solid", borderwidth=1, highlightthickness=2, highlightbackground="white", highlightcolor="black")
        self.output_viewer.pack(fill="both", expand=True)
        self.output_viewer.vbar.configure(troughcolor="white", bg="white", activebackground="#e0e0e0", borderwidth=0)

        self.viewer_menu = tk.Menu(self.output_viewer, tearoff=0, bg="white", font=("Arial", 10, "bold"))
        self.viewer_menu.add_command(label="🔍 Identify Source", command=self.run_hysilens)
        self.viewer_menu.add_command(label="🌐 Ask", command=self.run_anaxa)

        self.output_viewer.bind("<Button-3>", self.show_context_menu)
        self.output_viewer.bind("<Button-2>", self.show_context_menu)

        ordinal_frame = tk.Frame(self.center_panel, bg="white")
        ordinal_frame.pack(fill="x", pady=(5, 0))
        
        self._create_clickable_label(
            ordinal_frame, 
            "📤 Push Posts to Ordinal", 
            self.push_to_ordinal
        ).pack(side="right", padx=10)

        self._create_clickable_label(
            ordinal_frame, 
            "📥 Fetch Comments", 
            self.fetch_comments_from_ordinal
        ).pack(side="left", padx=10)
        
        # ==========================================
        #               RIGHT PANEL (DATA EXPLORER)
        # ==========================================
        explorer_header = tk.Frame(self.right_panel, bg="white")
        explorer_header.pack(fill="x", pady=(0, 5))
        
        self._place_chibi(explorer_header, side="left", size=(50, 50))
        tk.Label(explorer_header, text="Data Explorer", font=("Arial", 16, "bold"), bg="white").pack(side="left", padx=(5, 0))
        self._create_clickable_label(explorer_header, "🔄 Refresh", self.refresh_file_tree).pack(side="right", padx=10)

        tree_frame = tk.Frame(self.right_panel, bg="white", relief="solid", borderwidth=1)
        tree_frame.pack(fill="x", pady=(5, 10))
        
        self.file_tree = ttk.Treeview(tree_frame, height=6, show="tree", selectmode="browse")
        self.file_tree.pack(side="left", fill="both", expand=True)
        self.file_tree.bind("<<TreeviewSelect>>", self._on_file_select)
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="right", fill="y")

        search_frame = tk.Frame(self.right_panel, bg="white")
        search_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(search_frame, text="Search Doc:", bg="white", font=self.UI_FONT).pack(side="left")
        self.search_var = tk.StringVar()
        self.exp_search_entry = tk.Entry(search_frame, textvariable=self.search_var, font=self.UI_FONT, relief="solid", borderwidth=1, highlightthickness=1)
        self.exp_search_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        self.exp_search_entry.bind("<Return>", lambda e: self.search_document())
        self._create_clickable_label(search_frame, "🔍 Find", self.search_document).pack(side="right")

        # Intentionally unbolded per request: "...and Data Explorer"
        self.doc_explorer_viewer = scrolledtext.ScrolledText(self.right_panel, bg="#fdfdfd", fg="#333333", font=("Helvetica", 11), wrap="word", relief="solid", borderwidth=1, highlightthickness=1, state="disabled")
        self.doc_explorer_viewer.pack(fill="both", expand=True, pady=(0, 10))
        self.doc_explorer_viewer.vbar.configure(troughcolor="white", bg="white", activebackground="#e0e0e0", borderwidth=0)
        
        self.qa_frame = tk.LabelFrame(self.right_panel, text="Client Strategy Q&A", font=self.HEADER_FONT, padx=5, pady=5, bg="white", relief="flat", bd=0)
        self.qa_frame.pack(fill="x", pady=(5, 0))

        tk.Label(self.qa_frame, text="Paste Draft/Text to Evaluate (Optional):", bg="white", font=self.UI_FONT).pack(anchor="w")
        self.draft_text_input = tk.Text(self.qa_frame, height=4, font=self.UI_FONT, wrap="word", relief="solid", borderwidth=1, highlightthickness=2, highlightbackground="white", highlightcolor="black", insertbackground="black")
        self.draft_text_input.pack(fill="x", pady=(0, 10))

        tk.Label(self.qa_frame, text="Your Question (e.g., 'Why was this rejected?'):", bg="white", font=self.UI_FONT).pack(anchor="w", pady=(2, 2))
        self.question_input = tk.Entry(self.qa_frame, font=self.UI_FONT, relief="solid", borderwidth=1, highlightthickness=2, highlightbackground="white", highlightcolor="black")
        self.question_input.pack(fill="x", pady=(0, 10))

        self.query_btn = self._create_clickable_label(self.qa_frame, "🧠 Ask", self.run_cerydra_query)
        self.query_btn.pack(anchor="e", pady=5)
        
        self.refresh_file_tree()

        # ==========================================
        #               KEYBOARD SHORTCUTS
        # ==========================================
        self.root.bind("<Command-f>", self._handle_find_shortcut)
        self.root.bind("<Control-f>", self._handle_find_shortcut)

    def open_content_strategy_dialog(self):
        """Dialog to edit the content_strategy.txt file for a client."""
        company = self.company_var.get().strip()
        if not company:
            messagebox.showerror("Error", "Please enter a Company Keyword first!")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Content Strategy for {company}")
        dialog.geometry("700x500")
        dialog.configure(padx=15, pady=15, bg="white")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text="Edit content strategy, tone instructions, and style guidelines below.\n"
                 "This will be injected into the generation prompt as a critical directive.",
            bg="white", font=self.UI_FONT, wraplength=660, justify="left"
        ).pack(anchor="w", pady=(0, 8))

        text_area = scrolledtext.ScrolledText(
            dialog, height=18, font=("Courier", 10),
            relief="solid", borderwidth=1, highlightthickness=1, wrap="word"
        )
        text_area.pack(fill="both", expand=True, pady=(0, 10))

        # Load existing content_strategy.txt if it exists
        strategy_dir = str(P.content_strategy_dir(company))
        os.makedirs(strategy_dir, exist_ok=True)
        strategy_path = os.path.join(strategy_dir, "content_strategy.txt")
        
        existing_text = ""
        if os.path.exists(strategy_path):
            try:
                with open(strategy_path, "r", encoding="utf-8") as f:
                    existing_text = f.read()
            except Exception:
                existing_text = ""
        
        if existing_text:
            text_area.insert("1.0", existing_text)
        else:
            placeholder = (
                "# Content Strategy for " + company + "\n\n"
                "## Tone & Voice\n"
                "- [e.g., Conversational but authoritative]\n"
                "- [e.g., Avoid jargon unless explaining it]\n\n"
                "## Topics to Emphasize\n"
                "- [Key themes to focus on]\n\n"
                "## Topics to Avoid\n"
                "- [Things not to mention]\n\n"
                "## Style Notes\n"
                "- [Specific formatting preferences]\n"
                "- [Hook style preferences]\n"
            )
            text_area.insert("1.0", placeholder)

        def save_strategy():
            content = text_area.get("1.0", tk.END).strip()
            if not content:
                messagebox.showerror("Missing Content", "Please enter a content strategy.", parent=dialog)
                return

            try:
                with open(strategy_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"[Amphoreus] Saved content strategy to {strategy_path}")
                messagebox.showinfo("Saved", f"Content strategy saved to:\n{strategy_path}", parent=dialog)
                dialog.destroy()
                self.refresh_file_tree()
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save:\n{e}", parent=dialog)

        self._create_clickable_label(dialog, "💾 Save Content Strategy", save_strategy).pack(fill="x", pady=(4, 0))

    def open_screwllum_dialog(self):
        """
        Open the Screwllum content strategy generation popup.
        The agent researches all context autonomously via Ordinal analytics and
        Google Search — no manual input required beyond the company keyword.
        Output streams token-by-token into the popup window.
        """
        company = self.company_var.get().strip()
        if not company:
            messagebox.showerror("Error", "Please enter a Company Keyword first!")
            return

        # ── Build popup ───────────────────────────────────────────────────────
        popup = tk.Toplevel(self.root)
        popup.title(f"🧠 Content Strategy — {company}")
        popup.geometry("900x700")
        popup.configure(padx=15, pady=15, bg="white")
        popup.transient(self.root)

        # Header
        header_frame = tk.Frame(popup, bg="white")
        header_frame.pack(fill="x", pady=(0, 6))

        tk.Label(
            header_frame,
            text=f"Content Strategy: {company}",
            font=("Arial", 14, "bold"), bg="white",
        ).pack(side="left")

        tk.Label(
            header_frame,
            text="Agent researches follower data, post analytics & ICP automatically",
            font=("Arial", 9), fg="gray", bg="white",
        ).pack(side="left", padx=(10, 0))

        # Input rows
        field_font = ("Arial", 10, "bold")

        # Row 1 — goal + follower count + ICP %
        row1 = tk.Frame(popup, bg="white")
        row1.pack(fill="x", pady=(0, 4))

        tk.Label(row1, text="Primary Goal:", bg="white",
                 font=field_font).pack(side="left", padx=(0, 4))
        goal_var = tk.StringVar(value="pipeline")
        goal_menu = tk.OptionMenu(row1, goal_var, "pipeline", "brand", "mixed")
        goal_menu.config(bg="white", relief="solid", borderwidth=1, font=field_font,
                         highlightthickness=0, activebackground="#f0f0f0")
        goal_menu["menu"].config(bg="white", font=field_font)
        goal_menu.pack(side="left", padx=(0, 20))

        tk.Label(row1, text="LinkedIn Followers:", bg="white",
                 font=field_font).pack(side="left", padx=(0, 4))
        followers_var = tk.StringVar()
        tk.Entry(row1, textvariable=followers_var, font=field_font,
                 relief="solid", borderwidth=1, width=10).pack(side="left", padx=(0, 20))

        tk.Label(row1, text="% ICP among followers:", bg="white",
                 font=field_font).pack(side="left", padx=(0, 4))
        icp_var = tk.StringVar()
        tk.Entry(row1, textvariable=icp_var, font=field_font,
                 relief="solid", borderwidth=1, width=6).pack(side="left", padx=(0, 4))
        tk.Label(row1, text="%", bg="white", font=field_font).pack(side="left")

        # Row 2 — extra context
        row2 = tk.Frame(popup, bg="white")
        row2.pack(fill="x", pady=(0, 8))

        tk.Label(row2, text="Extra context (optional):",
                 bg="white", font=field_font).pack(side="left", padx=(0, 6))
        extra_var = tk.StringVar()
        tk.Entry(
            row2,
            textvariable=extra_var,
            font=field_font,
            relief="solid", borderwidth=1,
        ).pack(side="left", fill="x", expand=True)

        # Output text area
        output_frame = tk.Frame(popup, bg="white", relief="solid", bd=1)
        output_frame.pack(fill="both", expand=True, pady=(0, 8))

        output_text = scrolledtext.ScrolledText(
            output_frame,
            bg="#1a1a2e", fg="#e0e0e0",
            font=("Courier", 10),
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        output_text.pack(fill="both", expand=True, padx=2, pady=2)
        output_text.vbar.configure(
            troughcolor="#1a1a2e", bg="#444",
            activebackground="#666", borderwidth=0,
        )

        # Button row
        btn_frame = tk.Frame(popup, bg="white")
        btn_frame.pack(fill="x")

        run_lbl  = self._create_clickable_label(btn_frame, "▶ Start Generation", None)
        run_lbl.pack(side="left", padx=(0, 20))

        copy_lbl = self._create_clickable_label(btn_frame, "📋 Copy Output", None)
        copy_lbl.pack(side="left", padx=(0, 12))

        # "Open HTML Report" — starts disabled, enabled once the report is ready
        html_lbl = self._create_clickable_label(
            btn_frame, "🌐 Open HTML Report", None, fg="#555555"
        )
        html_lbl.config(state="disabled")
        html_lbl.pack(side="left", padx=(0, 12))

        # "Save Short-Term Strategy" — starts disabled, enabled after generation
        save_st_lbl = self._create_clickable_label(
            btn_frame, "💾 Save Short-Term Strategy", None, fg="#555555"
        )
        save_st_lbl.config(state="disabled")
        save_st_lbl.pack(side="left")

        close_lbl = self._create_clickable_label(
            btn_frame, "✖ Close", popup.destroy, fg="#cc0000"
        )
        close_lbl.pack(side="right")

        status_var = tk.StringVar(value="Ready — click ▶ to begin.")
        tk.Label(
            btn_frame, textvariable=status_var,
            bg="white", font=("Arial", 9, "bold"), fg="gray",
        ).pack(side="left", padx=20)

        # ── Save short-term strategy dialog ───────────────────────────────────
        def _open_save_short_term_dialog(short_term_text: str):
            """Open an editable dialog for the short-term section, then save."""
            edit_win = tk.Toplevel(popup)
            edit_win.title(f"💾 Short-Term Strategy — {company}")
            edit_win.geometry("800x560")
            edit_win.configure(padx=15, pady=15, bg="white")
            edit_win.transient(popup)
            edit_win.grab_set()

            tk.Label(
                edit_win,
                text="Edit the short-term strategy before saving:",
                font=("Arial", 11, "bold"), bg="white",
            ).pack(anchor="w", pady=(0, 6))

            tk.Label(
                edit_win,
                text=f"Will be saved to: memory/{company}/content_strategy/content_strategy.txt",
                font=("Arial", 9), fg="gray", bg="white",
            ).pack(anchor="w", pady=(0, 8))

            edit_area = scrolledtext.ScrolledText(
                edit_win,
                font=("Courier", 10),
                wrap="word",
                relief="solid",
                borderwidth=1,
                bg="#fafafa",
            )
            edit_area.pack(fill="both", expand=True, pady=(0, 10))
            edit_area.insert("1.0", short_term_text)

            save_status = tk.StringVar(value="")

            def _do_save():
                text_to_save = edit_area.get("1.0", tk.END).strip()
                try:
                    out_dir = P.content_strategy_dir(company)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / "content_strategy.txt"
                    out_file.write_text(text_to_save, encoding="utf-8")
                    save_status.set(f"✅ Saved to {out_file}")
                    save_btn.config(state="disabled")
                except Exception as exc:
                    save_status.set(f"❌ Save failed: {exc}")

            action_frame = tk.Frame(edit_win, bg="white")
            action_frame.pack(fill="x")

            save_btn = self._create_clickable_label(
                action_frame, "💾 Save to content_strategy.txt", _do_save
            )
            save_btn.pack(side="left", padx=(0, 20))

            self._create_clickable_label(
                action_frame, "✖ Close", edit_win.destroy, fg="#cc0000"
            ).pack(side="left")

            tk.Label(
                action_frame,
                textvariable=save_status,
                bg="white", font=("Arial", 9, "bold"), fg="#336633",
            ).pack(side="left", padx=16)

        # ── Thread-safe output callback ───────────────────────────────────────
        def _append(chunk: str):
            output_text.after(0, lambda c=chunk: (
                output_text.insert(tk.END, c),
                output_text.see(tk.END),
            ))

        # ── Copy handler ──────────────────────────────────────────────────────
        def _copy_output():
            content = output_text.get("1.0", tk.END).strip()
            popup.clipboard_clear()
            popup.clipboard_append(content)
            status_var.set("Copied to clipboard.")

        copy_lbl.bind("<Button-1>", lambda e: _copy_output())

        # ── Generation thread ─────────────────────────────────────────────────
        def _run():
            import webbrowser
            run_lbl.config(state="disabled")
            status_var.set("Running — agent is researching…")
            output_text.delete("1.0", tk.END)

            try:
                result = screwllum.run_programmatic(
                    client_name=company,
                    output_callback=_append,
                    primary_goal=goal_var.get(),
                    follower_count=followers_var.get().strip(),
                    icp_pct=icp_var.get().strip(),
                    extra_context=extra_var.get().strip(),
                )
                html_path    = result.get("html_path", "") if isinstance(result, dict) else ""
                strategy_txt = result.get("strategy", "")  if isinstance(result, dict) else ""
                status_var.set("Done." + (" HTML report ready." if html_path else ""))

                # Enable the Open HTML Report button if a file was produced
                if html_path:
                    def _open_html(path=html_path):
                        abs_path = Path(path).resolve()
                        webbrowser.open(abs_path.as_uri())
                    html_lbl.after(0, lambda: (
                        html_lbl.config(state="normal", fg="black"),
                        html_lbl.bind("<Button-1>", lambda e: _open_html()),
                    ))

                # Enable the Save Short-Term Strategy button
                if strategy_txt.strip():
                    short_term = screwllum.extract_short_term_section(strategy_txt)
                    def _open_save(st=short_term):
                        _open_save_short_term_dialog(st)
                    save_st_lbl.after(0, lambda: (
                        save_st_lbl.config(state="normal", fg="black"),
                        save_st_lbl.bind("<Button-1>", lambda e: _open_save()),
                    ))

            except Exception as exc:
                _append(f"\n\n❌ Error: {exc}\n")
                status_var.set(f"Error: {exc}")
            finally:
                run_lbl.after(0, lambda: run_lbl.config(state="normal"))

        def _start():
            import threading
            threading.Thread(target=_run, daemon=True).start()

        run_lbl.bind(
            "<Button-1>",
            lambda e: _start() if str(run_lbl["state"]) != "disabled" else None,
        )

    def open_past_posts_dialog(self):
        """Dialog to view and delete past posts used for redundancy checking."""
        company = self.company_var.get().strip()
        if not company:
            messagebox.showerror("Error", "Please enter a Company Keyword first!")
            return

        posts_dir = str(P.past_posts_dir(company))
        os.makedirs(posts_dir, exist_ok=True)
        index_path = os.path.join(posts_dir, "index.json")

        # Load existing posts
        existing_posts = []
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    existing_posts = json.load(f)
            except Exception:
                existing_posts = []

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Past Posts — {company}")
        dialog.geometry("800x650")
        dialog.configure(padx=15, pady=15, bg="white")
        dialog.transient(self.root)
        dialog.grab_set()

        # --- Header with count ---
        header_frame = tk.Frame(dialog, bg="white")
        header_frame.pack(fill="x", pady=(0, 8))
        count_var = tk.StringVar(value=f"{len(existing_posts)} past post(s) stored for redundancy checking")
        tk.Label(header_frame, textvariable=count_var, bg="white", font=self.HEADER_FONT).pack(side="left")

        tk.Label(
            dialog, 
            text="Posts are automatically saved after generation. They're used to detect redundant content.",
            bg="white", fg="gray", font=("Arial", 9)
        ).pack(anchor="w", pady=(0, 8))

        # --- Scrollable list of existing posts ---
        list_frame = tk.Frame(dialog, bg="white")
        list_frame.pack(fill="both", expand=True, pady=(0, 10))

        canvas = tk.Canvas(list_frame, bg="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        inner_frame = tk.Frame(canvas, bg="white")

        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._bind_mousewheel(canvas)

        check_vars = []  # list of (BooleanVar, index) for deletion

        def refresh_list():
            for widget in inner_frame.winfo_children():
                widget.destroy()
            check_vars.clear()

            if not existing_posts:
                tk.Label(inner_frame, text="No past posts yet. Posts are saved automatically after generation.",
                        bg="white", fg="gray", font=self.UI_FONT).pack(anchor="w", pady=5)
                return

            # Sort by date (most recent first)
            sorted_posts = sorted(existing_posts, key=lambda x: x.get("date", ""), reverse=True)

            for i, entry in enumerate(sorted_posts):
                row = tk.Frame(inner_frame, bg="white", relief="groove", bd=1)
                row.pack(fill="x", pady=2, padx=2)

                header_row = tk.Frame(row, bg="white")
                header_row.pack(fill="x")

                # Find original index
                original_idx = existing_posts.index(entry)
                var = tk.BooleanVar(value=False)
                check_vars.append((var, original_idx))
                tk.Checkbutton(header_row, variable=var, bg="white", activebackground="white").pack(side="left")

                date_str = entry.get("date", "unknown")
                theme_str = entry.get("theme", "")[:80] + "..." if len(entry.get("theme", "")) > 80 else entry.get("theme", "")
                tk.Label(header_row, text=f"[{date_str}] {theme_str}", bg="white", font=("Arial", 10, "bold"),
                        anchor="w").pack(side="left", fill="x", expand=True)

                # Post preview (first 200 chars)
                post_preview = entry.get("post", "")[:200].replace("\n", " ")
                if len(entry.get("post", "")) > 200:
                    post_preview += "..."
                tk.Label(row, text=post_preview, bg="#f8f8f8", font=("Arial", 9), fg="#555",
                        wraplength=700, justify="left", anchor="w", padx=5, pady=3).pack(fill="x")

            count_var.set(f"{len(existing_posts)} past post(s) stored for redundancy checking")

        refresh_list()

        # --- Delete selected ---
        def delete_selected():
            indices_to_remove = sorted([idx for var, idx in check_vars if var.get()], reverse=True)
            if not indices_to_remove:
                messagebox.showinfo("No Selection", "Select posts to delete first.", parent=dialog)
                return
            for idx in indices_to_remove:
                existing_posts.pop(idx)
            try:
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(existing_posts, f, indent=2, ensure_ascii=False)
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save: {e}", parent=dialog)
                return
            refresh_list()

        btn_frame = tk.Frame(dialog, bg="white")
        btn_frame.pack(fill="x", pady=(0, 10))
        self._create_clickable_label(btn_frame, "Delete Selected", delete_selected, fg="#cc0000").pack(side="left")
        self._create_clickable_label(btn_frame, "Clear All", lambda: clear_all(), fg="#cc0000").pack(side="left", padx=(8, 0))

        def clear_all():
            if not messagebox.askyesno("Confirm", f"Delete all {len(existing_posts)} past posts?", parent=dialog):
                return
            existing_posts.clear()
            try:
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(existing_posts, f, indent=2, ensure_ascii=False)
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save: {e}", parent=dialog)
                return
            refresh_list()

        # --- Add post manually ---
        tk.Frame(dialog, height=2, bd=1, relief="sunken", bg="#cccccc").pack(fill="x", pady=8)
        tk.Label(dialog, text="Manually add a past post (e.g., from LinkedIn):",
                bg="white", font=self.UI_FONT).pack(anchor="w", pady=(0, 4))

        add_text = scrolledtext.ScrolledText(dialog, height=6, font=("Courier", 10),
                                            relief="solid", borderwidth=1, highlightthickness=1)
        add_text.pack(fill="x", pady=(0, 8))

        def add_post():
            raw = add_text.get("1.0", tk.END).strip()
            if not raw:
                messagebox.showinfo("Empty", "Paste a post to add.", parent=dialog)
                return
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d")
            existing_posts.append({
                "post": raw,
                "theme": "(manually added)",
                "date": timestamp
            })
            try:
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(existing_posts, f, indent=2, ensure_ascii=False)
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save: {e}", parent=dialog)
                return
            add_text.delete("1.0", tk.END)
            refresh_list()

        self._create_clickable_label(dialog, "Add Post", add_post).pack(fill="x")

    def _handle_find_shortcut(self, event):
        """Routes Cmd/Ctrl + F to the correct search bar based on where focus currently is."""
        focused = self.root.focus_get()
        try:
            current = focused
            while current:
                if current == self.right_panel:
                    self.exp_search_entry.focus_set()
                    self.exp_search_entry.select_range(0, tk.END)
                    return "break"
                elif current == self.center_panel:
                    self.doc_search_entry.focus_set()
                    self.doc_search_entry.select_range(0, tk.END)
                    return "break"
                
                parent_name = current.winfo_parent()
                if not parent_name:
                    break
                current = current._nametowidget(parent_name)
        except Exception:
            pass
            
        # Default fallback: always send to the Document Viewer search bar
        if hasattr(self, 'doc_search_entry'):
            self.doc_search_entry.focus_set()
            self.doc_search_entry.select_range(0, tk.END)
        return "break"

    # --- ARCHIVE TRANSCRIPT HELPER ---
    def _archive_latest_transcript(self, target_dir):
        """Finds any existing 'latest.txt' or 'latest.docx' and renames them to 'Transcript {N}'."""
        for ext in [".txt", ".docx"]:
            latest_path = os.path.join(target_dir, f"latest{ext}")
            if os.path.exists(latest_path):
                i = 1
                while os.path.exists(os.path.join(target_dir, f"Transcript {i}.txt")) or os.path.exists(os.path.join(target_dir, f"Transcript {i}.docx")):
                    i += 1
                os.rename(latest_path, os.path.join(target_dir, f"Transcript {i}{ext}"))

    # --- UI HELPERS ---
    def _create_clickable_label(self, parent_frame, text, command, font=None, fg="black"):
        f = font or self.UI_FONT
        hf = (f[0], f[1] + 1, "bold") if isinstance(f, tuple) and len(f) >= 2 else self.HOVER_FONT
        lbl = tk.Label(parent_frame, text=text, font=f, fg=fg, bg="white", cursor="hand2")
        def on_enter(e):
            if str(lbl["state"]) != "disabled": lbl.config(font=hf, fg="#666666")
        def on_leave(e):
            if str(lbl["state"]) != "disabled": lbl.config(font=f, fg=fg)
        def on_click(e):
            if str(lbl["state"]) != "disabled": command()
        lbl.bind("<Enter>", on_enter)
        lbl.bind("<Leave>", on_leave)
        lbl.bind("<Button-1>", on_click)
        return lbl

    @staticmethod
    def _bind_mousewheel(canvas):
        """Bind trackpad / mousewheel scrolling to a canvas (works on macOS and Windows)."""
        _is_mac = platform.system() == "Darwin"

        def _scroll(event):
            if _is_mac:
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_enter(_e):
            canvas.bind_all("<MouseWheel>", _scroll)
            if not _is_mac:
                canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-3, "units"))
                canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(3, "units"))

        def _on_leave(_e):
            canvas.unbind_all("<MouseWheel>")
            if not _is_mac:
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _on_enter)
        canvas.bind("<Leave>", _on_leave)

    def _load_specific_image(self, path, size=(30, 30)):
        if os.path.exists(path):
            try:
                img = Image.open(path)
                img.thumbnail(size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.image_refs.append(photo)
                return photo
            except Exception as e: pass
        return None

    def _place_chibi(self, parent_frame, side="left", size=(45, 45)):
        if not self.chibi_images: return
        img_path = self.chibi_images.pop(0)
        self.chibi_images.append(img_path)
        try:
            img = Image.open(img_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_refs.append(photo)
            lbl = tk.Label(parent_frame, image=photo, bg="white")
            lbl.pack(side=side)
        except Exception as e: pass

    # --- RIGHT PANEL LOGIC ---
    def refresh_file_tree(self):
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
            
        tree_data = self.cerydra.get_file_tree()
        self._populate_tree("", tree_data)

    def _populate_tree(self, parent, tree_dict):
        sorted_items = sorted(tree_dict.items(), key=lambda item: (not isinstance(item[1], dict), item[0]))
        
        for key, value in sorted_items:
            if isinstance(value, dict):
                node = self.file_tree.insert(parent, "end", text=f"📁 {key}", open=False)
                self._populate_tree(node, value)
            else:
                self.file_tree.insert(parent, "end", text=f"📄 {key}", values=(value,))

    def _on_file_select(self, event):
        selected = self.file_tree.selection()
        if not selected: return
        
        item = selected[0]
        values = self.file_tree.item(item, "values")
        
        if values: 
            filepath = values[0]
            content = self.cerydra.read_document(filepath)
            
            self.doc_explorer_viewer.config(state="normal")
            self.doc_explorer_viewer.delete("1.0", tk.END)
            self.doc_explorer_viewer.insert("1.0", content)
            self.doc_explorer_viewer.config(state="disabled")

    def search_document(self):
        query = self.search_var.get().strip()
        self.doc_explorer_viewer.config(state="normal")
        self.cerydra.highlight_search_terms(self.doc_explorer_viewer, query)
        self.doc_explorer_viewer.config(state="disabled")

    def run_cerydra_query(self):
        company = self.company_var.get().strip()
        question = self.question_input.get().strip()
        draft = self.draft_text_input.get("1.0", tk.END).strip()

        if not company or not question:
            messagebox.showwarning("Missing Data", "Company Keyword and Question are required.")
            return

        self.query_btn.config(state="disabled", text="Analyzing...")
        self.console.insert(tk.END, f"\n[CERYDRA] Analyzing documents for '{company}' to answer your question...\n")
        self.console.see(tk.END)

        threading.Thread(target=self._cerydra_thread, args=(company, question, draft), daemon=True).start()

    def _cerydra_thread(self, company, question, draft):
        result = self.cerydra.query_documents(company, question, draft)
        self.root.after(0, lambda: self.show_cerydra_result(result))

    def show_cerydra_result(self, result_text):
        self.query_btn.config(state="normal", text="🧠 Ask Cerydra")
        
        popup = tk.Toplevel(self.root)
        popup.title("Cerydra Strategy Analysis")
        popup.geometry("650x450")
        popup.configure(padx=15, pady=15, bg="white")
        popup.transient(self.root)

        tk.Label(popup, text="🧠 Strategic Analysis", font=("Arial", 14, "bold"), bg="white", fg="#333").pack(anchor="w", pady=(0, 10))
        
        # Changed font to bold inside popup
        result_viewer = scrolledtext.ScrolledText(popup, bg="#f9f9f9", fg="#111", font=("Helvetica", 11, "bold"), wrap="word", relief="solid", borderwidth=1)
        result_viewer.pack(fill="both", expand=True)
        result_viewer.insert("1.0", result_text)
        result_viewer.config(state="disabled") 
        
        self._create_clickable_label(popup, "Close", popup.destroy).pack(pady=(10, 0))

    # --- LEFT PANEL: LOGIC FUNCTIONS ---

    def _get_target_dir(self, folder_type):
        company = self.company_var.get().strip()
        if not company:
            messagebox.showerror("Error", "Please enter a Company Keyword first!")
            return None
        P.ensure_dirs(company)
        if folder_type == "accepted": return str(P.accepted_dir(company))
        elif folder_type == "feedback": return str(P.feedback_dir(company))
        elif folder_type == "revisions": return str(P.revisions_dir(company))
        return str(P.transcripts_dir(company))
    
    def show_upload_dialog(self, folder_type):
        target_dir = self._get_target_dir(folder_type)
        if not target_dir: return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Add {folder_type.capitalize()} Documents")
        dialog.geometry("500x550")
        dialog.configure(padx=15, pady=15, bg="white")
        dialog.transient(self.root) 
        dialog.grab_set()           

        tk.Label(dialog, text="Option 1: Upload Existing Files", font=self.HEADER_FONT, bg="white").pack(anchor="w", pady=(0, 5))
        btn_browse = self._create_clickable_label(dialog, "Browse Files...", lambda: self._browse_files(folder_type, dialog))
        btn_browse.pack(anchor="w", pady=(0, 15))

        tk.Frame(dialog, height=2, bd=1, relief="sunken", bg="#cccccc").pack(fill="x", pady=10)

        tk.Label(dialog, text="Option 2: Paste Text Directly", font=self.HEADER_FONT, bg="white").pack(anchor="w", pady=(5, 5))
        
        # Modify the filename field logic depending on folder_type
        tk.Label(dialog, text="Filename (e.g., interview_1):", bg="white", font=self.UI_FONT).pack(anchor="w")
        filename_var = tk.StringVar()
        
        if folder_type == "base":
            filename_var.set("latest.txt")
            entry = tk.Entry(dialog, textvariable=filename_var, font=self.UI_FONT, relief="solid", borderwidth=1, highlightthickness=1)
            entry.config(state="disabled")
            entry.pack(fill="x", pady=(0, 10))
            tk.Label(dialog, text="*Base documents are automatically saved as 'latest.txt'", bg="white", fg="gray", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 5))
        else:
            tk.Entry(dialog, textvariable=filename_var, font=self.UI_FONT, relief="solid", borderwidth=1, highlightthickness=1).pack(fill="x", pady=(0, 10))

        tk.Label(dialog, text="Paste Text Below:", bg="white", font=self.UI_FONT).pack(anchor="w")
        text_area = scrolledtext.ScrolledText(dialog, height=12, font=self.UI_FONT, relief="solid", borderwidth=1, highlightthickness=1)
        text_area.pack(fill="both", expand=True, pady=(0, 10))

        def save_pasted_text():
            content = text_area.get(1.0, tk.END).strip()
            
            # Archive the old transcript if it's a base folder
            if folder_type == "base":
                self._archive_latest_transcript(target_dir)
                filename = "latest.txt"
            else:
                filename = filename_var.get().strip()
                if not filename:
                    messagebox.showerror("Missing Information", "Please provide a filename.", parent=dialog)
                    return
                if not filename.lower().endswith('.txt'):
                    filename += '.txt'

            if not content:
                messagebox.showerror("Missing Information", "Please provide text content.", parent=dialog)
                return
                
            os.makedirs(target_dir, exist_ok=True)
            filepath = os.path.join(target_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Saved pasted text to {folder_type} folder as {filename}.")
                self.refresh_file_tree()
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save text:\n{e}", parent=dialog)

        btn_save = self._create_clickable_label(dialog, "💾 Save Pasted Text", save_pasted_text)
        btn_save.pack(fill="x", pady=5)

    def _browse_files(self, folder_type, dialog_to_close=None):
        target_dir = self._get_target_dir(folder_type)
        if not target_dir: return
        
        files = filedialog.askopenfilenames(title=f"Select files for {folder_type}")
        if files:
            os.makedirs(target_dir, exist_ok=True)
            for file in files:
                try:
                    # If it's a base file, archive the existing "latest" and set the new file as "latest"
                    if folder_type == "base":
                        self._archive_latest_transcript(target_dir)
                        _, ext = os.path.splitext(file)
                        dest_path = os.path.join(target_dir, f"latest{ext}")
                        shutil.copyfile(file, dest_path)
                    else:
                        shutil.copy(file, target_dir)
                except Exception as e: pass
            
            self.refresh_file_tree()
            if dialog_to_close:
                dialog_to_close.destroy()
    
    def show_revision_dialog(self):
        company = self.company_var.get().strip()
        if not company:
            messagebox.showerror("Error", "Please enter a Company Keyword first!")
            return
        P.ensure_dirs(company)

        dlg = tk.Toplevel(self.root)
        dlg.title("Add Revision or Feedback")
        dlg.geometry("920x560")
        dlg.configure(bg="white")
        dlg.minsize(700, 420)
        dlg.transient(self.root)
        dlg.grab_set()

        outer = tk.Frame(dlg, bg="white")
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)

        tk.Label(
            outer, text="Pipeline Draft (Before) / Feedback",
            bg="white", font=self.HEADER_FONT,
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        tk.Label(
            outer, text="Revised Version (After) — leave blank for feedback only",
            bg="white", font=self.HEADER_FONT,
        ).grid(row=0, column=1, sticky="w", pady=(0, 4))

        left_t = scrolledtext.ScrolledText(
            outer, height=16, font=self.UI_FONT, wrap=tk.WORD,
            relief="solid", borderwidth=1,
        )
        left_t.grid(row=1, column=0, sticky="nsew", padx=(0, 6))

        right_t = scrolledtext.ScrolledText(
            outer, height=16, font=self.UI_FONT, wrap=tk.WORD,
            relief="solid", borderwidth=1,
        )
        right_t.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        tk.Label(
            outer, text="Notes (optional):", bg="white", font=self.UI_FONT,
        ).grid(row=2, column=0, sticky="w", pady=(8, 2), columnspan=2)

        notes_var = tk.StringVar()
        tk.Entry(
            outer, textvariable=notes_var, font=self.UI_FONT,
            relief="solid", borderwidth=1,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        def _next_filename(directory, prefix):
            os.makedirs(directory, exist_ok=True)
            existing = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".txt")]
            nums = []
            for f in existing:
                try:
                    nums.append(int(f.replace(prefix, "").replace(".txt", "")))
                except ValueError:
                    pass
            return f"{prefix}{max(nums, default=0) + 1}.txt"

        def save():
            left = left_t.get("1.0", tk.END).strip()
            right = right_t.get("1.0", tk.END).strip()
            notes = notes_var.get().strip() or "No notes provided."

            if not left and not right:
                messagebox.showerror("Missing Content", "Please paste text in at least the left panel.", parent=dlg)
                return
            if not left and right:
                messagebox.showerror("Missing Content", "The left panel (before/feedback) cannot be empty if the right panel has text.", parent=dlg)
                return

            try:
                if left and right:
                    rev_dir = str(P.revisions_dir(company))
                    fname = _next_filename(rev_dir, "revision-")
                    fpath = os.path.join(rev_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(f"=== PIPELINE DRAFT ===\n{left}\n\n")
                        f.write(f"=== REVISION NOTES ===\n{notes}\n\n")
                        f.write(f"=== REVISED VERSION ===\n{right}\n")
                    print(f"[Amphoreus] Saved revision to {fpath}")
                else:
                    fb_dir = str(P.feedback_dir(company))
                    fname = _next_filename(fb_dir, "feedback-")
                    fpath = os.path.join(fb_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(left)
                    print(f"[Amphoreus] Saved feedback to {fpath}")

                self.refresh_file_tree()
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save:\n{e}", parent=dlg)

        btn_row = tk.Frame(dlg, bg="white")
        btn_row.pack(fill=tk.X, padx=12, pady=(0, 12))
        self._create_clickable_label(btn_row, "Save", save).pack(side=tk.LEFT, padx=(0, 10))
        self._create_clickable_label(btn_row, "Close", dlg.destroy).pack(side=tk.LEFT)

    def _load_client_context(self) -> str:
        """Load transcripts + feedback + accepted posts for the current company."""
        company = self.company_var.get().strip()
        if not company:
            return ""
        parts = []
        transcripts = _read_local_context(str(P.transcripts_dir(company)), skip_files=[])
        if transcripts:
            parts.append(f"=== INTERVIEW TRANSCRIPTS ===\n{transcripts}")
        references = _read_local_context(str(P.references_dir(company)), skip_files=[])
        if references:
            parts.append(f"=== CLIENT REFERENCES ===\n{references}")
        accepted = _read_local_context(str(P.accepted_dir(company)), skip_files=[])
        if accepted:
            parts.append(f"=== APPROVED POSTS ===\n{accepted}")
        feedback = _read_local_context(str(P.feedback_dir(company)), skip_files=[])
        if feedback:
            parts.append(f"=== CLIENT FEEDBACK ===\n{feedback}")
        return "\n\n".join(parts)

    def show_context_menu(self, event):
        try:
            if self.output_viewer.tag_ranges(tk.SEL):
                self.viewer_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.viewer_menu.grab_release()

    def run_hysilens(self):
        try:
            selected_text = self.output_viewer.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return 

        company = self.company_var.get().strip()

        if not company:
            messagebox.showwarning("Missing Data", "Company Keyword is required to search source files.")
            return

        self.console.insert(tk.END, "\n[HYSILENS] Analyzing source for selected snippet...\n")
        self.console.see(tk.END)
        threading.Thread(target=self._hysilens_thread, args=(selected_text, company, "Gemini"), daemon=True).start()

    def _hysilens_thread(self, snippet, company, model):
        result = self.hysilens.find_source(snippet, company, model)
        self.root.after(0, lambda: self.show_hysilens_result(result))

    def show_hysilens_result(self, result_text):
        popup = tk.Toplevel(self.root)
        popup.title("Hysilens Source Analysis")
        popup.geometry("500x350")
        popup.configure(padx=15, pady=15, bg="white")
        popup.transient(self.root)

        tk.Label(popup, text="🔍 Hysilens Analysis", font=("Arial", 14, "bold"), bg="white", fg="#333").pack(anchor="w", pady=(0, 10))
        
        # Changed font to bold inside popup
        result_viewer = scrolledtext.ScrolledText(popup, bg="#f9f9f9", fg="#111", font=("Helvetica", 11, "bold"), wrap="word", relief="solid", borderwidth=1)
        result_viewer.pack(fill="both", expand=True)
        result_viewer.insert("1.0", result_text)
        result_viewer.config(state="disabled") 
        
        self._create_clickable_label(popup, "Close", popup.destroy).pack(pady=(10, 0))
    
    def run_anaxa(self):
        """Captures highlighted text and prompts the user for a web-search query."""
        try:
            # Grab the highlighted text from the output viewer
            selected_text = self.output_viewer.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            messagebox.showinfo("No Selection", "Please highlight text in the document viewer first.")
            return

        # Pop up a dialog box asking for the specific question
        user_query = simpledialog.askstring(
            "Anaxa Web Search", 
            "What would you like Anaxa to search the web for regarding this text?"
        )
        
        if not user_query:
            return # User cancelled the dialog

        self.console.insert(tk.END, f"\n[ANAXA] Searching the live web for: '{user_query}'...\n")
        self.console.see(tk.END)

        # Run in a thread to keep the UI responsive
        threading.Thread(target=self._anaxa_thread, args=(selected_text, user_query), daemon=True).start()

    def _anaxa_thread(self, text, query):
        """Executes the Anaxa API call and triggers the popup UI."""
        result = self.anaxa.query_with_search(highlighted_text=text, user_query=query)
        
        # Once the thread finishes, schedule the UI popup on the main thread
        self.root.after(0, lambda: self.show_anaxa_result(result, query))

    def show_anaxa_result(self, result_text, query):
        """Displays the Anaxa search results in a new popup window (like Hysilens)."""
        popup = tk.Toplevel(self.root)
        popup.title("Anaxa Web Search")
        popup.geometry("600x400")
        popup.configure(padx=15, pady=15, bg="white")
        popup.transient(self.root)

        # Display the user's query as the header
        tk.Label(popup, text=f"🌐 Query: {query}", font=("Arial", 12, "bold"), bg="white", fg="#333", wraplength=550, justify="left").pack(anchor="w", pady=(0, 10))
        
        # Changed font to bold inside popup
        result_viewer = scrolledtext.ScrolledText(popup, bg="#f9f9f9", fg="#111", font=("Helvetica", 11, "bold"), wrap="word", relief="solid", borderwidth=1)
        result_viewer.pack(fill="both", expand=True)
        result_viewer.insert("1.0", result_text)
        result_viewer.config(state="disabled") 
        
        self._create_clickable_label(popup, "Close", popup.destroy).pack(pady=(10, 0))

    def run_generation(self):
        client_name = self.client_name_var.get().strip()
        company = self.company_var.get().strip()
        action = self.action_var.get()

        if not client_name or not company:
            messagebox.showerror("Error", "Client Name and Company Keyword are required.")
            return

        self.run_btn.config(state="disabled", text="Running...")
        self.console.delete(1.0, tk.END)
        self.output_viewer.delete(1.0, tk.END)

        threading.Thread(
            target=self._process_thread,
            args=(client_name, company, action),
            daemon=True,
        ).start()

    def _process_thread(self, client_name, company, action):
        try:
            if action == "post":
                print(f"[Amphoreus] Starting post generation for {client_name}...\n")
                P.ensure_dirs(company)
                output_filepath = str(P.post_dir(company) / f"{company}_posts.md")
                generate_one_shot(client_name, company, output_filepath)
            else:
                print(f"[Amphoreus] Starting briefing for {client_name}...\n")
                generate_briefing(client_name, company)
        except Exception as e:
            print(f"\nAN ERROR OCCURRED: {e}")
        finally:
            self.run_btn.config(state="normal", text="🚀 RUN TASK")
            print("\n--- TASK FINISHED ---")
            self.root.after(0, self.load_output_to_viewer)
            self.root.after(0, self.refresh_file_tree)

    # ------------------------------------------------------------------
    # Cyrene rewrite dialogs
    # ------------------------------------------------------------------

    def open_single_post_rewrite_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Rewrite single post (Cyrene)")
        dlg.geometry("920x560")
        dlg.configure(bg="white")
        dlg.minsize(640, 420)
        dlg.transient(self.root)

        outer = tk.Frame(dlg, bg="white")
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=3)
        outer.rowconfigure(3, weight=1)

        tk.Label(
            outer,
            text="Post to rewrite",
            bg="white",
            font=self.HEADER_FONT,
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(
            outer,
            text="Rewrite instruction (optional)",
            bg="white",
            font=self.HEADER_FONT,
        ).grid(row=0, column=1, sticky="w", pady=(0, 4))

        post_t = scrolledtext.ScrolledText(
            outer,
            height=12,
            font=self.UI_FONT,
            wrap=tk.WORD,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="white",
            highlightcolor="black",
        )
        post_t.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        instr_t = scrolledtext.ScrolledText(
            outer,
            height=12,
            font=self.UI_FONT,
            wrap=tk.WORD,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="white",
            highlightcolor="black",
        )
        instr_t.insert(
            "1.0",
            "Leave blank for random stylistic noise, or type a specific instruction.",
        )
        instr_t.grid(row=1, column=1, sticky="nsew")

        tk.Label(
            outer,
            text="Theme (optional, preserved through rewrite)",
            bg="white",
            font=self.HEADER_FONT,
        ).grid(row=2, column=0, sticky="w", pady=(12, 4))

        tk.Label(
            outer,
            text="Image suggestion (optional, preserved through rewrite)",
            bg="white",
            font=self.HEADER_FONT,
        ).grid(row=2, column=1, sticky="w", pady=(12, 4))

        theme_t = tk.Entry(
            outer,
            font=self.UI_FONT,
            relief="solid",
            borderwidth=1,
        )
        theme_t.grid(row=3, column=0, sticky="ew", padx=(0, 8))

        img_sug_t = scrolledtext.ScrolledText(
            outer,
            height=3,
            font=self.UI_FONT,
            wrap=tk.WORD,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="white",
            highlightcolor="black",
        )
        img_sug_t.grid(row=3, column=1, sticky="nsew")

        ctx_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            outer, text="Load client context (transcripts, feedback, accepted posts)",
            variable=ctx_var, bg="white", activebackground="white", font=self.UI_FONT,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        btn_row = tk.Frame(dlg, bg="white")
        btn_row.pack(fill=tk.X, padx=12, pady=(0, 12))

        run_lbl = self._create_clickable_label(
            btn_row,
            "✨ Rewrite",
            lambda: self._submit_single_post_rewrite_from_dialog(post_t, instr_t, img_sug_t, theme_t, ctx_var, run_lbl, dlg),
        )
        run_lbl.pack(side=tk.LEFT, padx=(0, 10))

        self._create_clickable_label(btn_row, "Close", dlg.destroy).pack(side=tk.LEFT)

    def _submit_single_post_rewrite_from_dialog(self, post_t, instr_t, img_sug_t, theme_t, ctx_var, run_lbl, dlg):
        post_text = post_t.get("1.0", tk.END).strip()
        if not post_text:
            messagebox.showerror(
                "Error",
                "Paste the post you want rewritten.",
                parent=dlg,
            )
            return

        style_instruction = instr_t.get("1.0", tk.END).strip()
        if "Leave blank for random stylistic noise" in style_instruction:
            style_instruction = ""

        image_suggestion = img_sug_t.get("1.0", tk.END).strip()
        theme = theme_t.get().strip()
        client_context = self._load_client_context() if ctx_var.get() else ""

        run_lbl.config(state="disabled")

        def on_done():
            run_lbl.config(state="normal")

        threading.Thread(
            target=self._rewrite_pasted_thread,
            args=(post_text, style_instruction, image_suggestion, theme, client_context, on_done),
            daemon=True,
        ).start()

    def _rewrite_pasted_thread(self, post_text: str, style_instruction: str, image_suggestion: str = "", theme: str = "", client_context: str = "", on_done=None):
        try:
            print("\n[CYRENE] Rewriting pasted post...")
            cyrene = Cyrene()
            result = cyrene.rewrite_single_post(
                post_text=post_text,
                style_instruction=style_instruction,
                image_suggestion=image_suggestion,
                theme=theme,
                client_context=client_context,
            )
            print("[CYRENE] Pasted post rewrite complete.")
            
            self.root.after(
                0,
                lambda r=result: self._show_rewrite_result_popup(r),
            )
        except Exception as e:
            print(f"Rewrite (pasted) error: {e}")
            self.root.after(
                0,
                lambda err=str(e): messagebox.showerror(
                    "Rewrite failed",
                    err,
                    parent=self.root,
                ),
            )
        finally:
            if on_done:
                self.root.after(0, on_done)

    def _show_rewrite_result_popup(self, result: dict):
        popup = tk.Toplevel(self.root)
        popup.title("Rewritten post")
        popup.geometry("780x720")
        popup.configure(bg="white")
        popup.transient(self.root)

        final_post = result.get("final_post", "N/A")
        image_suggestion = result.get("image_suggestion", "")
        theme = result.get("theme", "")

        tk.Label(
            popup,
            text="Rewritten post",
            font=self.HEADER_FONT,
            bg="white",
        ).pack(anchor="w", padx=12, pady=(12, 4))

        body = scrolledtext.ScrolledText(
            popup,
            height=12,
            font=("Helvetica", 11),
            wrap="word",
            relief="solid",
            borderwidth=1,
            bg="#ffffff",
            fg="#333333",
        )
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        body.insert(tk.END, final_post + "\n")
        body.configure(state="disabled")

        btn_row = tk.Frame(popup, bg="white")
        btn_row.pack(fill="x", padx=12, pady=(0, 8))

        def copy_final():
            self.root.clipboard_clear()
            self.root.clipboard_append(final_post)
            self.root.update_idletasks()

        def show_cyrene_steps():
            detail = tk.Toplevel(popup)
            detail.title("Cyrene steps")
            detail.geometry("640x480")
            detail.configure(bg="white")
            t = scrolledtext.ScrolledText(
                detail,
                font=("Helvetica", 10),
                wrap="word",
                relief="solid",
                borderwidth=1,
            )
            t.pack(fill="both", expand=True, padx=10, pady=10)
            t.insert(
                tk.END,
                "**Step 1: Fact extraction**\n"
                + result.get("fact_extraction", "N/A")
                + "\n\n**Step 2: Style approach**\n"
                + result.get("style_analysis", "N/A")
                + "\n\n**Step 3: Rewrite strategy**\n"
                + result.get("strategy", "N/A")
                + "\n",
            )
            t.configure(state="disabled")
            self._create_clickable_label(detail, "Close", detail.destroy).pack(pady=(0, 10))

        self._create_clickable_label(btn_row, "Copy rewritten post", copy_final).pack(side="left", padx=(0, 8))
        self._create_clickable_label(btn_row, "Cyrene steps", show_cyrene_steps).pack(side="left", padx=(0, 8))
        self._create_clickable_label(btn_row, "Close", popup.destroy).pack(side="left")

        separator = ttk.Separator(popup, orient="horizontal")
        separator.pack(fill="x", padx=12, pady=(8, 8))

        ordinal_frame = tk.LabelFrame(
            popup,
            text="Push to Ordinal",
            bg="white",
            padx=8,
            pady=8,
            font=self.HEADER_FONT,
        )
        ordinal_frame.pack(fill="x", padx=12, pady=(0, 12))

        row1 = tk.Frame(ordinal_frame, bg="white")
        row1.pack(fill="x", pady=(0, 6))

        tk.Label(row1, text="Company:", bg="white", font=self.UI_FONT).pack(side="left")
        company_var = tk.StringVar(value=self.company_var.get().strip())
        company_entry = tk.Entry(row1, textvariable=company_var, width=20, font=self.UI_FONT)
        company_entry.pack(side="left", padx=(4, 12))

        tk.Label(row1, text="Publish date (YYYY-MM-DD HH:MM):", bg="white", font=self.UI_FONT).pack(side="left")
        default_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d 09:00")
        date_var = tk.StringVar(value=default_date)
        date_entry = tk.Entry(row1, textvariable=date_var, width=18, font=self.UI_FONT)
        date_entry.pack(side="left", padx=(4, 0))

        row2 = tk.Frame(ordinal_frame, bg="white")
        row2.pack(fill="x", pady=(0, 6))

        tk.Label(row2, text="Label:", bg="white", font=self.UI_FONT).pack(side="left")
        label_var = tk.StringVar()
        label_combo = ttk.Combobox(row2, textvariable=label_var, width=24, state="readonly")
        label_combo.pack(side="left", padx=(4, 12))

        tk.Label(row2, text="Approvers:", bg="white", font=self.UI_FONT).pack(side="left")
        approver_listbox = tk.Listbox(row2, selectmode=tk.MULTIPLE, height=3, width=30, font=("Arial", 10))
        approver_listbox.pack(side="left", padx=(4, 0))

        labels_cache = {"data": [], "ids": {}}
        users_cache = {"data": []}

        def fetch_ordinal_data():
            co = company_var.get().strip()
            if not co:
                return
            hyacinthia = Hyacinthia()
            labels = hyacinthia.get_labels(co)
            users = hyacinthia.get_users(co)
            labels_cache["data"] = labels
            labels_cache["ids"] = {lbl.get("name"): lbl.get("id") for lbl in labels}
            users_cache["data"] = users

            label_names = [lbl.get("name", "Unknown") for lbl in labels]
            label_combo["values"] = label_names
            if label_names:
                label_combo.current(0)

            approver_listbox.delete(0, tk.END)
            for u in users:
                name = f"{u.get('firstName', '')} {u.get('lastName', '')}".strip() or u.get("email", "Unknown")
                approver_listbox.insert(tk.END, name)

        fetch_btn = self._create_clickable_label(row2, "Fetch labels/approvers",
            lambda: threading.Thread(target=fetch_ordinal_data, daemon=True).start(),
            font=("Arial", 10))
        fetch_btn.pack(side="left", padx=(8, 0))

        if theme or image_suggestion:
            row3 = tk.Frame(ordinal_frame, bg="white")
            row3.pack(fill="x", pady=(0, 6))
            tk.Label(row3, text="Theme & image suggestion (will be added as comment):", bg="white", font=self.UI_FONT).pack(anchor="w")
            meta_text = scrolledtext.ScrolledText(row3, height=3, font=("Arial", 10), wrap="word", bg="#f9f9f9")
            meta_text.pack(fill="x")
            parts = []
            if theme:
                parts.append(f"Theme: {theme}")
            if image_suggestion:
                parts.append(f"Image suggestion: {image_suggestion}")
            meta_text.insert(tk.END, "\n".join(parts))
            meta_text.configure(state="disabled")

        def push_to_ordinal():
            co = company_var.get().strip()
            if not co:
                messagebox.showerror("Error", "Company keyword required.", parent=popup)
                return
            try:
                pub_dt = datetime.strptime(date_var.get().strip(), "%Y-%m-%d %H:%M")
            except ValueError:
                messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD HH:MM", parent=popup)
                return

            selected_label_name = label_var.get()
            label_ids = []
            if selected_label_name and selected_label_name in labels_cache["ids"]:
                label_ids = [labels_cache["ids"][selected_label_name]]

            selected_approver_indices = list(approver_listbox.curselection())
            approvals = []
            for idx in selected_approver_indices:
                if idx < len(users_cache["data"]):
                    user = users_cache["data"][idx]
                    approvals.append({"userId": user.get("id")})

            def do_push():
                hyacinthia = Hyacinthia()
                res = hyacinthia.push_single_post(
                    company_keyword=co,
                    content=final_post,
                    publish_date=pub_dt,
                    status="ForReview",
                    label_ids=label_ids if label_ids else None,
                    approvals=approvals if approvals else None,
                )
                if not res["success"]:
                    popup.after(0, lambda: messagebox.showerror("Ordinal Error", res["error"], parent=popup))
                    return

                post_id = res["post_id"]
                post_url = res["url"]

                if theme or image_suggestion:
                    comment_parts = []
                    if theme:
                        comment_parts.append(f"**Theme:** {theme}")
                    if image_suggestion:
                        comment_parts.append(f"**Image Suggestion:**\n{image_suggestion}")
                    comment_msg = "\n\n".join(comment_parts)
                    cmt_res = hyacinthia.create_comment(co, post_id, comment_msg)
                    if not cmt_res["success"]:
                        print(f"[ORDINAL] Comment creation failed: {cmt_res['error']}")

                popup.after(0, lambda: messagebox.showinfo(
                    "Success",
                    f"Post pushed to Ordinal!\nID: {post_id}\nURL: {post_url}",
                    parent=popup,
                ))

            threading.Thread(target=do_push, daemon=True).start()

        push_btn = self._create_clickable_label(ordinal_frame, "Push to Ordinal", push_to_ordinal, fg="#4CAF50")
        push_btn.pack(anchor="w", pady=(4, 0))

    def _parse_posts_from_output_file(self, filepath: str) -> list:
        """
        Parse posts from a Stelle output file ({company}_posts.md).

        Returns list of dicts with keys: post, theme, image_suggestion, why_post, origin
        Extracts the "Final Post" section (post-fact-check), plus metadata.
        """
        import re

        if not os.path.exists(filepath):
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        post_blocks = re.split(r'^## Post \d+:', content, flags=re.MULTILINE)

        parsed = []
        for block in post_blocks[1:]:
            block = block.strip()

            hook = block.split("\n")[0].strip() if block else ""

            origin_match = re.search(r'\*\*Origin:\*\*\s*(.+)', block)
            origin = origin_match.group(1).strip() if origin_match else ""

            final_match = re.search(
                r'### Final Post\s*\n(.*?)(?=###|\Z)', block, re.DOTALL,
            )
            if final_match:
                post_text = final_match.group(1).strip()
            else:
                draft_match = re.search(
                    r'### Draft\s*\n(.*?)(?=###|\Z)', block, re.DOTALL,
                )
                post_text = draft_match.group(1).strip() if draft_match else ""

            why_match = re.search(
                r'### Why Post\s*\n(.*?)(?=###|---|\Z)', block, re.DOTALL,
            )
            why_post = why_match.group(1).strip() if why_match else ""

            img_match = re.search(
                r'### Image Suggestion\s*\n(.*?)(?=###|---|\Z)', block, re.DOTALL,
            )
            image_suggestion = img_match.group(1).strip() if img_match else ""

            if post_text:
                parsed.append({
                    "post": post_text,
                    "theme": hook,
                    "origin": origin,
                    "image_suggestion": image_suggestion,
                    "why_post": why_post,
                })

        return parsed[:MAX_BATCH_POSTS]

    def open_batch_post_rewrite_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Rewrite multiple posts — Cyrene (max {MAX_BATCH_POSTS})")
        dlg.geometry("820x640")
        dlg.configure(bg="white")
        dlg.minsize(640, 480)
        dlg.transient(self.root)

        outer = tk.Frame(dlg, bg="white")
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        top = tk.Frame(outer, bg="white")
        top.pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            top,
            text="How many posts?",
            bg="white",
            font=self.UI_FONT,
        ).pack(side=tk.LEFT, padx=(0, 8))
        count_var = tk.IntVar(value=3)
        spin = tk.Spinbox(
            top,
            from_=1,
            to=MAX_BATCH_POSTS,
            textvariable=count_var,
            width=4,
            font=self.UI_FONT,
        )
        spin.pack(side=tk.LEFT)
        tk.Label(
            top,
            text="(Each slot below must contain text.)",
            bg="white",
            fg="gray",
            font=("Arial", 9),
        ).pack(side=tk.LEFT, padx=(12, 0))

        load_row = tk.Frame(outer, bg="white")
        load_row.pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            load_row,
            text="Or load from output file:",
            bg="white",
            font=self.UI_FONT,
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        # post_entries will be defined later, so we need a reference holder
        post_entries_ref = {"entries": None, "sync_visible": None, "count_var": count_var, "scroll_region": None}

        def load_from_output():
            company = self.company_var.get().strip()
            if not company:
                messagebox.showerror("Error", "Please enter a Company Keyword first.", parent=dlg)
                return

            filename = f"{company}_posts.md"
            filepath = str(P.post_dir(company) / filename)
            
            if not os.path.exists(filepath):
                messagebox.showerror("Error", f"File not found:\n{filepath}", parent=dlg)
                return
            
            parsed = self._parse_posts_from_output_file(filepath)
            if not parsed:
                messagebox.showwarning("Warning", "No posts found in file.", parent=dlg)
                return
            
            entries = post_entries_ref["entries"]
            if not entries:
                return
            
            # Update count and sync visible
            n = min(len(parsed), MAX_BATCH_POSTS)
            count_var.set(n)
            if post_entries_ref["sync_visible"]:
                post_entries_ref["sync_visible"]()
            
            # Load data into entries
            for i, p in enumerate(parsed):
                if i >= len(entries):
                    break
                lf, st, img_entry, theme_entry = entries[i]
                
                # Clear and set post text
                st.delete("1.0", tk.END)
                st.insert("1.0", p.get("post", ""))
                
                # Clear and set theme
                theme_entry.delete(0, tk.END)
                theme_entry.insert(0, p.get("theme", ""))
                
                # Clear and set image suggestion
                img_entry.delete(0, tk.END)
                img_entry.insert(0, p.get("image_suggestion", ""))
            
            messagebox.showinfo("Loaded", f"Loaded {n} post(s) from {filename}", parent=dlg)

        self._create_clickable_label(load_row, "Load posts", load_from_output).pack(side=tk.LEFT)

        scroll_wrap = tk.Frame(outer, bg="white")
        scroll_wrap.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        canvas = tk.Canvas(scroll_wrap, bg="white", highlightthickness=0)
        vsb = ttk.Scrollbar(scroll_wrap, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="white")
        inner_win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _canvas_width(event):
            canvas.itemconfigure(inner_win, width=event.width)

        inner.bind("<Configure>", _scroll_region)
        canvas.bind("<Configure>", _canvas_width)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=vsb.set)

        self._bind_mousewheel(canvas)

        post_entries: list[tuple] = []
        for i in range(MAX_BATCH_POSTS):
            lf = tk.LabelFrame(
                inner,
                text=f"Post {i + 1}",
                bg="white",
                padx=6,
                pady=6,
                font=self.UI_FONT,
            )
            st = scrolledtext.ScrolledText(
                lf,
                height=5,
                font=self.UI_FONT,
                wrap=tk.WORD,
                relief="solid",
                borderwidth=1,
                highlightthickness=1,
                highlightbackground="white",
                highlightcolor="black",
            )
            st.pack(fill=tk.BOTH, expand=True)
            
            meta_row = tk.Frame(lf, bg="white")
            meta_row.pack(fill=tk.X, pady=(4, 0))
            
            tk.Label(meta_row, text="Theme:", bg="white", fg="gray", font=("Arial", 9)).pack(side=tk.LEFT)
            theme_entry = tk.Entry(meta_row, font=("Arial", 10), width=25)
            theme_entry.pack(side=tk.LEFT, padx=(2, 10))
            
            tk.Label(meta_row, text="Image suggestion:", bg="white", fg="gray", font=("Arial", 9)).pack(side=tk.LEFT)
            img_entry = tk.Entry(meta_row, font=("Arial", 10), width=40)
            img_entry.pack(side=tk.LEFT, padx=(2, 0), fill=tk.X, expand=True)
            
            post_entries.append((lf, st, img_entry, theme_entry))

        def sync_visible(*_args):
            try:
                n = int(count_var.get())
            except (tk.TclError, ValueError):
                n = 1
            n = max(1, min(n, MAX_BATCH_POSTS))
            count_var.set(n)
            for idx, (lf, _st, _img_entry, _theme_entry) in enumerate(post_entries):
                if idx < n:
                    lf.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
            else:
                    lf.pack_forget()
            _scroll_region()

        spin.configure(command=sync_visible)
        sync_visible()

        # Store references for load_from_output
        post_entries_ref["entries"] = post_entries
        post_entries_ref["sync_visible"] = sync_visible
        post_entries_ref["scroll_region"] = _scroll_region

        tk.Label(
            outer,
            text="Batch rewrite instruction (applies to the whole set; Cyrene sees every post at once)",
            bg="white",
            font=self.HEADER_FONT,
        ).pack(anchor="w", pady=(0, 4))
        instr_t = scrolledtext.ScrolledText(
            outer,
            height=4,
            font=self.UI_FONT,
            wrap=tk.WORD,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="white",
            highlightcolor="black",
        )
        instr_t.insert(
            "1.0",
            "e.g. Remove redundant lines across posts, align tone, or leave blank for varied stylistic noise.",
        )
        instr_t.pack(fill=tk.X, pady=(0, 8))

        batch_ctx_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            outer, text="Load client context (transcripts, feedback, accepted posts)",
            variable=batch_ctx_var, bg="white", activebackground="white", font=self.UI_FONT,
        ).pack(anchor="w", pady=(0, 8))

        btn_row = tk.Frame(dlg, bg="white")
        btn_row.pack(fill=tk.X, padx=12, pady=(0, 12))

        run_lbl = self._create_clickable_label(
            btn_row,
            "✨ Rewrite batch",
            lambda: self._submit_batch_post_rewrite_from_dialog(
                count_var, post_entries, instr_t, batch_ctx_var, run_lbl, dlg
            ),
        )
        run_lbl.pack(side=tk.LEFT, padx=(0, 10))

        self._create_clickable_label(btn_row, "Close", dlg.destroy).pack(side=tk.LEFT)

    def _submit_batch_post_rewrite_from_dialog(self, count_var, post_entries, instr_t, batch_ctx_var, run_lbl, dlg):
        try:
            n = int(count_var.get())
        except (tk.TclError, ValueError):
            n = 1
        n = max(1, min(n, MAX_BATCH_POSTS))
        posts = [post_entries[i][1].get("1.0", tk.END).strip() for i in range(n)]
        image_suggestions = [post_entries[i][2].get().strip() for i in range(n)]
        themes = [post_entries[i][3].get().strip() for i in range(n)]
        if not any(posts):
            messagebox.showerror(
                "Error",
                "Paste at least one post in the active slots.",
                parent=dlg,
            )
            return
        if any(not p for p in posts):
            messagebox.showerror(
                "Error",
                f"All {n} active post slots must contain text (or lower the count).",
                parent=dlg,
            )
            return

        style_instruction = instr_t.get("1.0", tk.END).strip()
        if "e.g. Remove redundant lines across posts" in style_instruction:
            style_instruction = ""

        client_context = self._load_client_context() if batch_ctx_var.get() else ""

        run_lbl.config(state="disabled")

        def on_done():
            run_lbl.config(state="normal")

        threading.Thread(
            target=self._rewrite_batch_pasted_thread,
            args=(posts, style_instruction, image_suggestions, themes, client_context, on_done),
            daemon=True,
        ).start()

    def _rewrite_batch_pasted_thread(self, posts: list, style_instruction: str, image_suggestions: list = None, themes: list = None, client_context: str = "", on_done=None):
        try:
            print(f"\n[CYRENE] Rewriting pasted batch ({len(posts)} posts)...")
            cyrene = Cyrene()
            result = cyrene.rewrite_post_batch(
                posts=posts,
                style_instruction=style_instruction,
                image_suggestions=image_suggestions,
                themes=themes,
                client_context=client_context,
            )
            print("[CYRENE] Batch rewrite complete.")
            
            self.root.after(
                0,
                lambda r=result: self._show_batch_rewrite_result_popup(r),
            )
        except ValueError as e:
            self.root.after(
                0,
                lambda err=str(e): messagebox.showerror(
                    "Batch rewrite",
                    err,
                    parent=self.root,
                ),
            )
        except Exception as e:
            print(f"Rewrite (batch paste) error: {e}")
            self.root.after(
                0,
                lambda err=str(e): messagebox.showerror(
                    "Rewrite failed",
                    err,
                    parent=self.root,
                ),
            )
        finally:
            if on_done:
                self.root.after(0, on_done)

    def _show_batch_rewrite_result_popup(self, result: dict):
        popup = tk.Toplevel(self.root)
        popup.title("Rewritten posts (batch)")
        popup.geometry("820x780")
        popup.configure(bg="white")
        popup.transient(self.root)

        notes = result.get("batch_coordinator_notes", "N/A")
        posts = result.get("posts") or []

        tk.Label(
            popup,
            text="Batch coordinator notes",
            font=self.HEADER_FONT,
            bg="white",
        ).pack(anchor="w", padx=12, pady=(12, 4))
        notes_w = scrolledtext.ScrolledText(
            popup,
            height=4,
            font=("Helvetica", 10),
            wrap="word",
            relief="solid",
            borderwidth=1,
            bg="#fafafa",
            fg="#333333",
        )
        notes_w.pack(fill=tk.X, padx=12, pady=(0, 8))
        notes_w.insert(tk.END, notes + "\n")
        notes_w.configure(state="disabled")

        pick_row = tk.Frame(popup, bg="white")
        pick_row.pack(fill=tk.X, padx=12, pady=(0, 4))
        tk.Label(pick_row, text="Post:", bg="white", font=self.UI_FONT).pack(side=tk.LEFT)
        post_labels = [f"Post {p.get('index', i + 1)}" for i, p in enumerate(posts)]
        sel_var = tk.StringVar(value=post_labels[0] if post_labels else "")
        combo = ttk.Combobox(
            pick_row,
            textvariable=sel_var,
            values=post_labels,
            state="readonly",
            width=14,
            font=self.UI_FONT,
        )
        if post_labels:
            combo.pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(
            popup,
            text="Rewritten post",
            font=("Arial", 11, "bold"),
            bg="white",
        ).pack(anchor="w", padx=12, pady=(8, 4))

        body = scrolledtext.ScrolledText(
            popup,
            height=10,
            font=("Helvetica", 11),
            wrap="word",
            relief="solid",
            borderwidth=1,
            bg="#ffffff",
            fg="#333333",
        )
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        def current_post_dict():
            if not posts:
                return {}
            try:
                idx = post_labels.index(sel_var.get())
            except ValueError:
                idx = 0
            return posts[idx] if idx < len(posts) else posts[0]

        def refresh_body(*_args):
            body.configure(state="normal")
            body.delete("1.0", tk.END)
            pd = current_post_dict()
            body.insert(tk.END, (pd.get("final_post") or "N/A") + "\n")
            body.configure(state="disabled")

        if post_labels:
            combo.bind("<<ComboboxSelected>>", refresh_body)
        refresh_body()

        btn_row = tk.Frame(popup, bg="white")
        btn_row.pack(fill="x", padx=12, pady=(0, 8))

        def copy_current():
            pd = current_post_dict()
            text = (pd.get("final_post") or "").strip()
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update_idletasks()

        def copy_all_finals():
            parts = []
            for p in posts:
                i = p.get("index", "?")
                fp = (p.get("final_post") or "").strip()
                parts.append(f"--- Post {i} ---\n{fp}")
            blob = "\n\n".join(parts)
            self.root.clipboard_clear()
            self.root.clipboard_append(blob)
            self.root.update_idletasks()

        def show_steps_for_current():
            pd = current_post_dict()
            detail = tk.Toplevel(popup)
            detail.title(f"Cyrene steps — post {pd.get('index', '?')}")
            detail.geometry("640x480")
            detail.configure(bg="white")
            t = scrolledtext.ScrolledText(
                detail,
                font=("Helvetica", 10),
                wrap="word",
                relief="solid",
                borderwidth=1,
            )
            t.pack(fill="both", expand=True, padx=10, pady=10)
            t.insert(
                tk.END,
                "**Step 1: Fact extraction**\n"
                + str(pd.get("fact_extraction", "N/A"))
                + "\n\n**Step 2: Style approach**\n"
                + str(pd.get("style_analysis", "N/A"))
                + "\n\n**Step 3: Rewrite strategy**\n"
                + str(pd.get("strategy", "N/A"))
                + "\n",
            )
            t.configure(state="disabled")
            self._create_clickable_label(detail, "Close", detail.destroy).pack(pady=(0, 10))

        self._create_clickable_label(btn_row, "Copy this post", copy_current).pack(side=tk.LEFT, padx=(0, 8))
        self._create_clickable_label(btn_row, "Copy all finals", copy_all_finals).pack(side=tk.LEFT, padx=(0, 8))
        self._create_clickable_label(btn_row, "Cyrene steps (this post)", show_steps_for_current).pack(side=tk.LEFT, padx=(0, 8))
        self._create_clickable_label(btn_row, "Close", popup.destroy).pack(side=tk.LEFT)

        separator = ttk.Separator(popup, orient="horizontal")
        separator.pack(fill="x", padx=12, pady=(8, 8))

        per_post_ordinal = [{"label": "", "approver_idxs": set()} for _ in posts]
        _batch_ord_nav_prev = [None]

        ordinal_frame = tk.LabelFrame(
            popup,
            text="Push current post to Ordinal",
            bg="white",
            padx=8,
            pady=8,
            font=self.HEADER_FONT,
        )
        ordinal_frame.pack(fill="x", padx=12, pady=(0, 12))

        row1 = tk.Frame(ordinal_frame, bg="white")
        row1.pack(fill="x", pady=(0, 6))

        tk.Label(row1, text="Company:", bg="white", font=self.UI_FONT).pack(side="left")
        company_var = tk.StringVar(value=self.company_var.get().strip())
        company_entry = tk.Entry(row1, textvariable=company_var, width=20, font=self.UI_FONT)
        company_entry.pack(side="left", padx=(4, 12))

        tk.Label(row1, text="Publish date (YYYY-MM-DD HH:MM):", bg="white", font=self.UI_FONT).pack(side="left")
        default_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d 09:00")
        date_var = tk.StringVar(value=default_date)
        date_entry = tk.Entry(row1, textvariable=date_var, width=18, font=self.UI_FONT)
        date_entry.pack(side="left", padx=(4, 0))

        row2 = tk.Frame(ordinal_frame, bg="white")
        row2.pack(fill="x", pady=(0, 6))

        tk.Label(row2, text="Label:", bg="white", font=self.UI_FONT).pack(side="left")
        label_var = tk.StringVar()
        label_combo = ttk.Combobox(row2, textvariable=label_var, width=20, state="readonly")
        label_combo.pack(side="left", padx=(4, 4))

        tk.Label(row2, text="Approvers:", bg="white", font=self.UI_FONT).pack(side="left")
        approver_listbox = tk.Listbox(row2, selectmode=tk.MULTIPLE, height=3, width=26, font=("Arial", 10))
        approver_listbox.pack(side="left", padx=(4, 0))

        tk.Label(
            row2,
            text="(per post — switch Post above)",
            bg="white",
            fg="gray",
            font=("Arial", 9),
        ).pack(side="left", padx=(4, 0))

        labels_cache = {"data": [], "ids": {}}
        users_cache = {"data": []}

        def _current_post_index():
            if not post_labels:
                return -1
            try:
                return post_labels.index(sel_var.get())
            except ValueError:
                return 0

        def fetch_ordinal_data():
            co = company_var.get().strip()
            if not co:
                return
            hyacinthia = Hyacinthia()
            labels_data = hyacinthia.get_labels(co)
            users = hyacinthia.get_users(co)
            labels_cache["data"] = labels_data
            labels_cache["ids"] = {lbl.get("name"): lbl.get("id") for lbl in labels_data}
            users_cache["data"] = users

            label_names = [lbl.get("name", "Unknown") for lbl in labels_data]
            label_combo["values"] = label_names
            if label_names:
                for st in per_post_ordinal:
                    if not st["label"]:
                        st["label"] = label_names[0]

            approver_listbox.delete(0, tk.END)
            for u in users:
                name = f"{u.get('firstName', '')} {u.get('lastName', '')}".strip() or u.get("email", "Unknown")
                approver_listbox.insert(tk.END, name)

            cur = _current_post_index()
            if 0 <= cur < len(per_post_ordinal):
                st = per_post_ordinal[cur]
                label_var.set(st["label"])
                approver_listbox.selection_clear(0, tk.END)
                for i in st["approver_idxs"]:
                    if 0 <= i < approver_listbox.size():
                        approver_listbox.selection_set(i)

        fetch_btn = self._create_clickable_label(row2, "Fetch labels/approvers",
            lambda: threading.Thread(target=fetch_ordinal_data, daemon=True).start(),
            font=("Arial", 10))
        fetch_btn.pack(side="left", padx=(8, 0))

        row3 = tk.Frame(ordinal_frame, bg="white")
        row3.pack(fill="x", pady=(0, 6))
        tk.Label(row3, text="Post metadata (auto-comment on Ordinal):", bg="white", font=self.UI_FONT).pack(anchor="w")
        meta_text = scrolledtext.ScrolledText(row3, height=4, font=("Arial", 10), wrap="word", bg="#f9f9f9")
        meta_text.pack(fill="x")

        def refresh_post_metadata(*_args):
            pd = current_post_dict()
            post_theme = pd.get("theme", "")
            img_sug = pd.get("image_suggestion", "")
            wp = pd.get("why_post", "")
            meta_text.configure(state="normal")
            meta_text.delete("1.0", tk.END)
            parts = []
            if wp:
                parts.append(f"Why post: {wp}")
            if post_theme:
                parts.append(f"Theme: {post_theme}")
            if img_sug:
                parts.append(f"Image suggestion: {img_sug}")
            if parts:
                meta_text.insert(tk.END, "\n".join(parts))
            else:
                meta_text.insert(tk.END, "(No theme or image suggestion for this post)")
            meta_text.configure(state="disabled")

        def _on_batch_post_pick(*_args):
            new_idx = _current_post_index()
            if new_idx < 0 or new_idx >= len(per_post_ordinal):
                return
            prev = _batch_ord_nav_prev[0]
            if prev is not None and 0 <= prev < len(per_post_ordinal):
                per_post_ordinal[prev]["label"] = label_var.get()
                per_post_ordinal[prev]["approver_idxs"] = set(approver_listbox.curselection())
            st = per_post_ordinal[new_idx]
            label_var.set(st["label"])
            approver_listbox.selection_clear(0, tk.END)
            for i in st["approver_idxs"]:
                if 0 <= i < approver_listbox.size():
                    approver_listbox.selection_set(i)
            _batch_ord_nav_prev[0] = new_idx
            refresh_body()
            refresh_post_metadata()

        if post_labels:
            combo.bind("<<ComboboxSelected>>", _on_batch_post_pick)
            _on_batch_post_pick()
        else:
            refresh_post_metadata()

        def push_to_ordinal():
            co = company_var.get().strip()
            if not co:
                messagebox.showerror("Error", "Company keyword required.", parent=popup)
                return
            try:
                pub_dt = datetime.strptime(date_var.get().strip(), "%Y-%m-%d %H:%M")
            except ValueError:
                messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD HH:MM", parent=popup)
                return

            pd = current_post_dict()
            final_post = pd.get("final_post", "").strip()
            image_suggestion = pd.get("image_suggestion", "").strip()
            post_theme = pd.get("theme", "").strip()
            why_post = pd.get("why_post", "").strip()

            if not final_post:
                messagebox.showerror("Error", "No post content to push.", parent=popup)
                return

            pidx = _current_post_index() if post_labels else -1
            if 0 <= pidx < len(per_post_ordinal):
                per_post_ordinal[pidx]["label"] = label_var.get()
                per_post_ordinal[pidx]["approver_idxs"] = set(approver_listbox.curselection())

            st = per_post_ordinal[pidx] if 0 <= pidx < len(per_post_ordinal) else {"label": "", "approver_idxs": set()}
            selected_label_name = st["label"]
            label_ids = []
            if selected_label_name and selected_label_name in labels_cache["ids"]:
                label_ids = [labels_cache["ids"][selected_label_name]]
            approvals = []
            for aidx in sorted(st["approver_idxs"]):
                if aidx < len(users_cache["data"]):
                    approvals.append({"userId": users_cache["data"][aidx].get("id")})

            def do_push():
                hyacinthia = Hyacinthia()
                res = hyacinthia.push_single_post(
                    company_keyword=co,
                    content=final_post,
                    publish_date=pub_dt,
                    status="ForReview",
                    label_ids=label_ids if label_ids else None,
                    approvals=approvals if approvals else None,
                )
                if not res["success"]:
                    popup.after(0, lambda: messagebox.showerror("Ordinal Error", res["error"], parent=popup))
                    return

                post_id = res["post_id"]
                post_url = res["url"]

                comment_parts = []
                if why_post:
                    comment_parts.append(f"**Why Post:**\n{why_post}")
                if post_theme:
                    comment_parts.append(f"**Theme:** {post_theme}")
                if image_suggestion:
                    comment_parts.append(f"**Image Suggestion:**\n{image_suggestion}")
                if comment_parts:
                    cmt_res = hyacinthia.create_comment(co, post_id, "\n\n".join(comment_parts))
                    if not cmt_res["success"]:
                        print(f"[ORDINAL] Comment creation failed: {cmt_res['error']}")

                popup.after(0, lambda: messagebox.showinfo(
                    "Success",
                    f"Post pushed to Ordinal!\nID: {post_id}\nURL: {post_url}",
                    parent=popup,
                ))

            threading.Thread(target=do_push, daemon=True).start()

        push_btn = self._create_clickable_label(ordinal_frame, "Push current post to Ordinal", push_to_ordinal, fg="#4CAF50")
        push_btn.pack(anchor="w", pady=(4, 0))

    def _load_specific_file(self, filepath):
        self.output_viewer.delete(1.0, tk.END)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                self.output_viewer.insert(tk.END, content)
            except Exception as e:
                self.output_viewer.insert(tk.END, f"Error reading file:\n{e}")
        else:
            self.output_viewer.insert(tk.END, "File not found.")

    def load_output_to_viewer(self):
        company = self.company_var.get().strip()
        action = self.action_var.get()
        if not company:
            return

        if action == "brief":
            filename = f"{company}_briefing.md"
            out_dir = P.brief_dir(company)
        else:
            filename = f"{company}_posts.md"
            out_dir = P.post_dir(company)

        filepath = str(out_dir / filename)
        self.output_viewer.delete(1.0, tk.END)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                self.output_viewer.insert(tk.END, content)
                print(f"Loaded {os.path.basename(filepath)} into viewer.")
            except Exception as e: 
                self.output_viewer.insert(tk.END, f"Error reading file:\n{e}")
        else: 
            self.output_viewer.insert(tk.END, f"Waiting for output...\n(Could not find {os.path.basename(filepath)})")

    def search_doc_viewer(self):
        """Searches the document viewer and highlights all matching text."""
        query = self.doc_search_var.get()
        
        # Remove any existing search highlights
        self.output_viewer.tag_remove("search_highlight", "1.0", tk.END)
        
        if not query:
            return
            
        start_pos = "1.0"
        first_match = None
        
        while True:
            # nocase=True makes the search case-insensitive
            start_pos = self.output_viewer.search(query, start_pos, nocase=True, stopindex=tk.END)
            if not start_pos:
                break
                
            if not first_match:
                first_match = start_pos
                
            # Calculate the end position based on query length
            end_pos = f"{start_pos}+{len(query)}c"
            
            # Apply the highlight tag
            self.output_viewer.tag_add("search_highlight", start_pos, end_pos)
            
            # Move start_pos forward to continue searching the rest of the document
            start_pos = end_pos
            
        # Configure the visual style of the highlight tag
        self.output_viewer.tag_config("search_highlight", background="yellow", foreground="black")
        
        if first_match:
            # Scroll the viewer so the first match is visible
            self.output_viewer.see(first_match)
        else:
            messagebox.showinfo("Search", f"No matches found for '{query}'.")

    def clear_doc_search(self):
        """Clears the search bar and removes all highlights."""
        self.doc_search_var.set("")
        self.output_viewer.tag_remove("search_highlight", "1.0", tk.END)
    
    # =============================================================================
    # DEPRECATED: Text Gradient / Prompt Graph Tuning Methods
    # These features have been removed. Methods kept as stubs for backwards compat.
    # =============================================================================
    def open_gradient_dialog(self):
        """DEPRECATED: Prompt graph tuning has been removed."""
        messagebox.showinfo("Deprecated", "Prompt graph tuning has been removed from this version.")

    def open_paste_feedback_dialog(self):
        """DEPRECATED: Prompt graph tuning has been removed."""
        messagebox.showinfo("Deprecated", "Prompt graph tuning has been removed from this version.")

    def rollback_gradient_step(self):
        """DEPRECATED: Prompt graph tuning has been removed."""
        messagebox.showinfo("Deprecated", "Prompt graph tuning has been removed from this version.")

    def push_to_ordinal(self):
        """
        Parse generated posts from the selected model's output file and open a
        per-post push dialog with labels, approvers, date, and auto-comment
        for theme + image suggestion.
        """
        company = self.company_var.get().strip()

        if not company:
            messagebox.showerror("Error", "Please enter a Company Keyword first.")
            return

        filename = f"{company}_posts.md"
        filepath = os.path.join(str(P.post_dir(company)), filename)
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"Could not find generated posts.\nLooked for: {filepath}")
            return
                
        parsed = self._parse_posts_from_output_file(filepath)
        if not parsed:
            messagebox.showwarning("Warning", f"No posts found in {filename}")
            return

        # Per-post Ordinal options (label name + approver listbox indices)
        per_post_ordinal = [{"label": "", "approver_idxs": set()} for _ in range(len(parsed))]
        _ordinal_nav_prev_idx = [None]

        # ---- build dialog ----
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Push to Ordinal — {filename} ({len(parsed)} posts)")
        dlg.geometry("980x720")
        dlg.configure(bg="white")
        dlg.minsize(800, 550)
        dlg.transient(self.root)

        outer = tk.Frame(dlg, bg="white")
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # -- post navigator --
        nav = tk.Frame(outer, bg="white")
        nav.pack(fill=tk.X, pady=(0, 6))

        post_labels = [f"Post {i+1}: {p.get('theme','')[:60]}" for i, p in enumerate(parsed)]
        nav_var = tk.StringVar(value=post_labels[0])
        nav_combo = ttk.Combobox(nav, textvariable=nav_var, values=post_labels,
                                  state="readonly", width=70, font=self.UI_FONT)
        nav_combo.pack(side=tk.LEFT)

        def _prev():
            idx = nav_combo.current()
            if idx > 0:
                nav_combo.current(idx - 1)
                _on_nav_change()

        def _next():
            idx = nav_combo.current()
            if idx < len(parsed) - 1:
                nav_combo.current(idx + 1)
                _on_nav_change()

        self._create_clickable_label(nav, "◀", _prev).pack(side=tk.LEFT, padx=(8, 2))
        self._create_clickable_label(nav, "▶", _next).pack(side=tk.LEFT)

        # -- post preview --
        preview = scrolledtext.ScrolledText(outer, height=12, font=("Arial", 10), wrap=tk.WORD,
                                            relief="solid", borderwidth=1, state="normal")
        preview.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        # -- Ordinal settings --
        settings = tk.LabelFrame(outer, text="Ordinal Settings", bg="white", padx=8, pady=8,
                                  font=self.HEADER_FONT)
        settings.pack(fill=tk.X, pady=(0, 6))

        row1 = tk.Frame(settings, bg="white")
        row1.pack(fill=tk.X, pady=(0, 6))

        tk.Label(row1, text="Company:", bg="white", font=self.UI_FONT).pack(side=tk.LEFT)
        co_var = tk.StringVar(value=company)
        tk.Entry(row1, textvariable=co_var, width=18, font=self.UI_FONT).pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(row1, text="Publish date:", bg="white", font=self.UI_FONT).pack(side=tk.LEFT)
        default_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d 09:00")
        date_var = tk.StringVar(value=default_date)
        tk.Entry(row1, textvariable=date_var, width=18, font=self.UI_FONT).pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(row1, text="Freq:", bg="white", font=self.UI_FONT).pack(side=tk.LEFT)
        freq_var = tk.IntVar(value=12)
        tk.Radiobutton(row1, text="12/mo", variable=freq_var, value=12, bg="white",
                        font=("Arial", 9)).pack(side=tk.LEFT)
        tk.Radiobutton(row1, text="8/mo", variable=freq_var, value=8, bg="white",
                        font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 8))

        row2 = tk.Frame(settings, bg="white")
        row2.pack(fill=tk.X, pady=(0, 6))

        tk.Label(row2, text="Label:", bg="white", font=self.UI_FONT).pack(side=tk.LEFT)
        label_var = tk.StringVar()
        label_combo = ttk.Combobox(row2, textvariable=label_var, width=22, state="readonly")
        label_combo.pack(side=tk.LEFT, padx=(4, 4))

        tk.Label(row2, text="Approvers:", bg="white", font=self.UI_FONT).pack(side=tk.LEFT)
        approver_listbox = tk.Listbox(row2, selectmode=tk.MULTIPLE, height=3, width=28,
                                       font=("Arial", 10))
        approver_listbox.pack(side=tk.LEFT, padx=(4, 0))

        tk.Label(
            row2,
            text="(each post — pick post above, set label/approvers, then next)",
            bg="white",
            fg="gray",
            font=("Arial", 9),
        ).pack(side=tk.LEFT, padx=(6, 0))

        labels_cache = {"data": [], "ids": {}}
        users_cache = {"data": []}

        def fetch_ordinal_data():
            co = co_var.get().strip()
            if not co:
                return
            hyacinthia = Hyacinthia()
            labels_data = hyacinthia.get_labels(co)
            users = hyacinthia.get_users(co)
            labels_cache["data"] = labels_data
            labels_cache["ids"] = {lbl.get("name"): lbl.get("id") for lbl in labels_data}
            users_cache["data"] = users

            label_names = [lbl.get("name", "Unknown") for lbl in labels_data]
            label_combo["values"] = label_names
            if label_names:
                for st in per_post_ordinal:
                    if not st["label"]:
                        st["label"] = label_names[0]

            approver_listbox.delete(0, tk.END)
            for u in users:
                name = f"{u.get('firstName', '')} {u.get('lastName', '')}".strip() or u.get("email", "Unknown")
                approver_listbox.insert(tk.END, name)

            cur = nav_combo.current()
            if cur >= 0 and cur < len(per_post_ordinal):
                st = per_post_ordinal[cur]
                label_var.set(st["label"])
                approver_listbox.selection_clear(0, tk.END)
                for i in st["approver_idxs"]:
                    if 0 <= i < approver_listbox.size():
                        approver_listbox.selection_set(i)

        self._create_clickable_label(row2, "Fetch labels/approvers",
                  lambda: threading.Thread(target=fetch_ordinal_data, daemon=True).start(),
                  font=("Arial", 10)).pack(side=tk.LEFT, padx=(8, 0))

        # -- theme / image suggestion meta --
        meta_frame = tk.Frame(settings, bg="white")
        meta_frame.pack(fill=tk.X, pady=(0, 4))
        tk.Label(meta_frame, text="Why post, theme & image suggestion (auto-comment):", bg="white",
                 font=self.UI_FONT).pack(anchor="w")
        meta_text = scrolledtext.ScrolledText(meta_frame, height=3, font=("Arial", 10),
                                               wrap="word", bg="#f9f9f9")
        meta_text.pack(fill=tk.X)

        # -- refresh preview, metadata, and per-post label/approver widgets when nav changes --
        def _on_nav_change(*_):
            new_idx = nav_combo.current()
            if new_idx < 0 or new_idx >= len(parsed):
                return
            prev = _ordinal_nav_prev_idx[0]
            if prev is not None and 0 <= prev < len(per_post_ordinal):
                per_post_ordinal[prev]["label"] = label_var.get()
                per_post_ordinal[prev]["approver_idxs"] = set(approver_listbox.curselection())
            st = per_post_ordinal[new_idx]
            label_var.set(st["label"])
            approver_listbox.selection_clear(0, tk.END)
            for i in st["approver_idxs"]:
                if 0 <= i < approver_listbox.size():
                    approver_listbox.selection_set(i)
            _ordinal_nav_prev_idx[0] = new_idx

            p = parsed[new_idx]
            preview.configure(state="normal")
            preview.delete("1.0", tk.END)
            preview.insert("1.0", p.get("post", ""))
            preview.configure(state="disabled")

            meta_text.configure(state="normal")
            meta_text.delete("1.0", tk.END)
            parts = []
            if p.get("why_post"):
                parts.append(f"Why post: {p['why_post']}")
            if p.get("theme"):
                parts.append(f"Theme: {p['theme']}")
            if p.get("image_suggestion"):
                parts.append(f"Image suggestion: {p['image_suggestion']}")
            meta_text.insert(tk.END, "\n".join(parts) if parts else "(none)")
            meta_text.configure(state="disabled")

        nav_combo.bind("<<ComboboxSelected>>", _on_nav_change)
        _on_nav_change()

        # -- push checkboxes (select which posts to push) --
        sel_frame = tk.Frame(outer, bg="white")
        sel_frame.pack(fill=tk.X, pady=(0, 6))
        tk.Label(sel_frame, text="Select posts to push:", bg="white", font=self.UI_FONT).pack(side=tk.LEFT)

        post_checks: list[tk.BooleanVar] = []
        checks_inner = tk.Frame(sel_frame, bg="white")
        checks_inner.pack(side=tk.LEFT, padx=(8, 0))
        for i in range(len(parsed)):
            v = tk.BooleanVar(value=True)
            post_checks.append(v)
            tk.Checkbutton(checks_inner, text=str(i + 1), variable=v, bg="white",
                           font=("Arial", 9)).pack(side=tk.LEFT)

        def _sel_all():
            for v in post_checks:
                v.set(True)

        def _sel_none():
            for v in post_checks:
                v.set(False)

        self._create_clickable_label(sel_frame, "All", _sel_all, font=("Arial", 9)).pack(side=tk.LEFT, padx=(12, 2))
        self._create_clickable_label(sel_frame, "None", _sel_none, font=("Arial", 9)).pack(side=tk.LEFT)

        # -- action buttons --
        btn_frame = tk.Frame(outer, bg="white")
        btn_frame.pack(fill=tk.X, pady=(4, 0))

        status_lbl = tk.Label(btn_frame, text="", bg="white", fg="gray", font=("Arial", 10))
        status_lbl.pack(side=tk.LEFT, padx=(0, 12))

        def _do_push():
            co = co_var.get().strip()
            if not co:
                messagebox.showerror("Error", "Company keyword required.", parent=dlg)
                return
            try:
                start_dt = datetime.strptime(date_var.get().strip(), "%Y-%m-%d %H:%M")
            except ValueError:
                messagebox.showerror("Error", "Invalid date. Use YYYY-MM-DD HH:MM", parent=dlg)
                return

            selected = [i for i, v in enumerate(post_checks) if v.get()]
            if not selected:
                messagebox.showwarning("Nothing selected", "Select at least one post.", parent=dlg)
                return

            cur_nav = nav_combo.current()
            if 0 <= cur_nav < len(per_post_ordinal):
                per_post_ordinal[cur_nav]["label"] = label_var.get()
                per_post_ordinal[cur_nav]["approver_idxs"] = set(approver_listbox.curselection())

            posts_per_month = freq_var.get()
            hyacinthia = Hyacinthia()
            dates = hyacinthia._compute_publish_dates(start_dt, len(selected), posts_per_month)

            def thread_push():
                ok_count = 0
                for seq, post_idx in enumerate(selected):
                    p = parsed[post_idx]
                    post_content = p.get("post", "")
                    theme = p.get("theme", "")
                    img_sug = p.get("image_suggestion", "")
                    why_post = p.get("why_post", "")

                    pub_dt = dates[seq] if seq < len(dates) else dates[-1]
                    status_lbl.config(text=f"Pushing post {post_idx + 1}...")

                    st = per_post_ordinal[post_idx]
                    selected_label_name = st["label"]
                    label_ids = []
                    if selected_label_name and selected_label_name in labels_cache["ids"]:
                        label_ids = [labels_cache["ids"][selected_label_name]]
                    approvals = []
                    for aidx in sorted(st["approver_idxs"]):
                        if aidx < len(users_cache["data"]):
                            approvals.append({"userId": users_cache["data"][aidx].get("id")})

                    res = hyacinthia.push_single_post(
                        company_keyword=co,
                        content=post_content,
                        publish_date=pub_dt,
                        status="ForReview",
                        label_ids=label_ids if label_ids else None,
                        approvals=approvals if approvals else None,
                    )
                    if not res["success"]:
                        print(f"[ORDINAL] Post {post_idx + 1} failed: {res['error']}")
                        continue

                    post_id = res["post_id"]
                    ok_count += 1

                    comment_parts = []
                    if why_post:
                        comment_parts.append(f"**Why Post:**\n{why_post}")
                    if theme:
                        comment_parts.append(f"**Theme:** {theme}")
                    if img_sug:
                        comment_parts.append(f"**Image Suggestion:**\n{img_sug}")
                    if comment_parts:
                        hyacinthia.create_comment(co, post_id, "\n\n".join(comment_parts))

                schedule_desc = "Mon/Wed/Thu" if posts_per_month == 12 else "Tue/Thu"
                dlg.after(0, lambda: status_lbl.config(text=f"Done — {ok_count}/{len(selected)} pushed."))
                dlg.after(0, lambda: messagebox.showinfo(
                    "Ordinal Push Complete",
                    f"Pushed {ok_count} of {len(selected)} posts.\nSchedule: {schedule_desc}",
                    parent=dlg,
                ))

            threading.Thread(target=thread_push, daemon=True).start()

        self._create_clickable_label(btn_frame, "Push Selected Posts", _do_push, fg="#4CAF50").pack(side=tk.RIGHT)
        self._create_clickable_label(btn_frame, "Close", dlg.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    def fetch_comments_from_ordinal(self):
        """
        Prompts the user for a date, fetches standard and inline comments from Ordinal,
        and saves them to the client's temporary directory.
        """
        company = self.company_var.get().strip()
        
        if not company:
            messagebox.showerror("Error", "Please enter a Company Keyword first.")
            return

        date_str = simpledialog.askstring(
            "Fetch Comments", 
            "Fetch comments for posts scheduled on/after (YYYY-MM-DD):", 
            initialvalue="2026-03-01"
        )
        
        if not date_str:
            return

        publish_date_min = f"{date_str}T00:00:00.000Z"
        
        try:
            hyacinthia = Hyacinthia()
            comments_data = hyacinthia.get_recent_comments(company_keyword=company, publish_date_min=publish_date_min)
            
            standard_comments = comments_data.get("standard_comments", [])
            inline_comments = comments_data.get("inline_comments", [])
            
            if not standard_comments and not inline_comments:
                messagebox.showinfo("Fetch Comments", "No comments found for the given criteria.")
                return

            tmp_dir = str(P.tmp_dir(company))
            os.makedirs(tmp_dir, exist_ok=True)
            
            json_file = os.path.join(tmp_dir, "fetched_comments.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(comments_data, f, indent=4)
                
            txt_file = os.path.join(tmp_dir, "fetched_comments.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"--- ORDINAL CLIENT FEEDBACK (Fetched: {publish_date_min}) ---\n\n")
                
                if standard_comments:
                    f.write("=== GENERAL COMMENTS ===\n")
                    for c in standard_comments:
                        user = c.get("user", {})
                        author = f"{user.get('firstName', '')} {user.get('lastName', '')}".strip()
                        f.write(f"[{author}]: {c.get('message')}\n")
                    f.write("\n")
                    
                if inline_comments:
                    f.write("=== INLINE COMMENTS (Text-Anchored) ===\n")
                    for ic in inline_comments:
                        # Include whether the feedback was resolved, but capture the anchored text
                        resolved_tag = "[RESOLVED] " if ic.get("resolved") else ""
                        highlighted = ic.get("highlightedText", "Unknown text")
                        
                        f.write(f"{resolved_tag}Highlight: \"{highlighted}\"\n")
                        
                        # Inline comments contain arrays of replies, format them as threads
                        for reply in ic.get("replies", []):
                            user = reply.get("user", {})
                            author = f"{user.get('firstName', '')} {user.get('lastName', '')}".strip()
                            f.write(f"  -> [{author}]: {reply.get('message')}\n")
                        f.write("\n")
                    
            messagebox.showinfo(
                "Fetch Comments Successful", 
                f"Successfully fetched {len(standard_comments)} standard and {len(inline_comments)} inline comment(s)!\n\nSaved to:\n{tmp_dir}"
            )
            
        except Exception as e:
            messagebox.showerror("System Error", f"An error occurred while fetching comments: {str(e)}")

if __name__ == "__main__":
    import traceback
    P.MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    P.PRODUCTS_ROOT.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "backend", "static", "images"), exist_ok=True)
    try:
        root = tk.Tk()
        print("[Amphoreus] Tk root created")
        app = AmphoreusExperiment(root)
        print("[Amphoreus] App initialized — launching GUI")
        root.mainloop()
    except Exception:
        traceback.print_exc()