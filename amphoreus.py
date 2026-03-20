import os
from dotenv import load_dotenv

load_dotenv()

import shutil
import threading
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from PIL import Image, ImageTk

from phainon import generate_iterative_linkedin_posts
from aglaea import generate_briefing
from anaxa import Anaxa
from cerydra import Cerydra
from cyrene import Cyrene
from hysilens import Hysilens

class PrintLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
    def flush(self):
        pass

class AmphoreusExperiment:
    def __init__(self, root):
        self.root = root
        self.root.title("The Amphoreus Experiment")
        self.root.geometry("1500x950") 
        self.root.configure(padx=10, pady=10, bg="white")

        # --- STATE ---
        self.current_generation_data = None
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
        static_dir = "./static/images"
        
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

        v1_frame = tk.Frame(self.left_panel, bg="white")
        v1_frame.pack(fill="x", pady=0)
        self._place_chibi(v1_frame, side="left", size=(45, 45))
        tk.Label(v1_frame, text="\"A picture of grace and elegance.\"", font=("Georgia", 10, "bold", "italic"), fg="#888888", bg="white").pack(side="left", padx=5)

        frame_files = tk.LabelFrame(self.left_panel, text="2. Upload Documents", padx=5, pady=5, bg="white", relief="flat", bd=0, font=self.HEADER_FONT)
        frame_files.pack(fill="x", pady=2)
        
        tk.Label(frame_files, text="Make sure Company Keyword is filled out first.", fg="gray", bg="white", font=("Arial", 9, "bold")).pack(pady=(0, 4))
        frame_files.pack_propagate(True)
        
        self._create_clickable_label(frame_files, "Add Base Files (Transcripts/Profile)", lambda: self.show_upload_dialog("base")).pack(fill="x", pady=8)
        self._create_clickable_label(frame_files, "Add Accepted Posts (Optional)", lambda: self.show_upload_dialog("accepted")).pack(fill="x", pady=8)
        self._create_clickable_label(frame_files, "Add Client Feedback (Optional)", lambda: self.show_upload_dialog("feedback")).pack(fill="x", pady=8)

        v2_frame = tk.Frame(self.left_panel, bg="white")
        v2_frame.pack(fill="x", pady=0)
        self._place_chibi(v2_frame, side="right", size=(45, 45))
        tk.Label(v2_frame, text="\"What do we have here?\"", font=("Georgia", 10, "bold", "italic"), fg="#888888", bg="white").pack(side="right", padx=5)

        frame_gen = tk.LabelFrame(self.left_panel, text="3. Generation", padx=5, pady=5, bg="white", relief="flat", bd=0, font=self.HEADER_FONT)
        frame_gen.pack(fill="x", pady=2)

        self.action_var = tk.StringVar(value="brief")
        
        rb_frame1 = tk.Frame(frame_gen, bg="white")
        rb_frame1.pack(fill="x", anchor="w")
        tk.Radiobutton(rb_frame1, text="Generate Aglaea and Castorice's Briefing?", variable=self.action_var, value="brief", bg="white", activebackground="white", highlightbackground="white", font=self.UI_FONT).pack(side="left")
        if self.aglaea_img:
            tk.Label(rb_frame1, image=self.aglaea_img, bg="white").pack(side="left", padx=(5, 0))

        rb_frame2 = tk.Frame(frame_gen, bg="white")
        rb_frame2.pack(fill="x", anchor="w")
        tk.Radiobutton(rb_frame2, text="Commence Flame-Chase Journey (12 Coreflames)?", variable=self.action_var, value="post", bg="white", activebackground="white", highlightbackground="white", font=self.UI_FONT).pack(side="left")
        if self.phainon_img:
            tk.Label(rb_frame2, image=self.phainon_img, bg="white").pack(side="left", padx=(5, 0))

        tk.Label(frame_gen, text="Model Choice:", bg="white", font=self.UI_FONT).pack(anchor="w", pady=(6, 2))
        self.model_var = tk.StringVar(value="All (Ensemble)")
        model_options = ["All (Ensemble)", "Gemini 3.1 Pro", "GPT-5", "Claude Opus 4.6"]
        
        model_menu = tk.OptionMenu(frame_gen, self.model_var, *model_options)
        model_menu.config(bg="white", activebackground="#f0f0f0", relief="solid", borderwidth=1, highlightbackground="white", highlightthickness=0, font=self.UI_FONT)
        model_menu["menu"].config(bg="white", font=self.UI_FONT)
        model_menu.pack(anchor="w", pady=(0, 6), fill="x")

        self.run_btn = self._create_clickable_label(frame_gen, "🚀 RUN AMPHOREUS EXPERIMENT", self.run_generation)
        self.run_btn.pack(fill="x", pady=8)

        # --- 4. STYLE TRANSFER (REWRITE) ---
        frame_rewrite = tk.LabelFrame(self.left_panel, text="4. Stylistic Rewrite with Cyrene", padx=5, pady=5, bg="white", relief="flat", bd=0, font=self.HEADER_FONT)
        frame_rewrite.pack(fill="x", pady=2)
        
        tk.Label(frame_rewrite, text="Style Instruction (Optional):", bg="white", font=self.UI_FONT).pack(anchor="w", pady=(2, 2))
        
        self.style_prompt_text = tk.Text(frame_rewrite, height=5, relief="solid", borderwidth=1, font=self.UI_FONT, bg="white", fg="black", highlightthickness=2, highlightbackground="white", highlightcolor="black", insertbackground="black")
        self.style_prompt_text.insert("1.0", "Leave blank for random stylistic noise, or type a specific instruction.")
        self.style_prompt_text.pack(anchor="w", fill="x", pady=(0, 6))
        
        self.rewrite_btn = self._create_clickable_label(frame_rewrite, "✨ Rewrite Posts", self.run_rewrite)
        self.rewrite_btn.pack(fill="x", pady=8)

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
        self.viewer_menu.add_command(label="🔍 Identify Source with Hysilens", command=self.run_hysilens)
        self.viewer_menu.add_command(label="🌐 Ask Anaxa", command=self.run_anaxa)

        self.output_viewer.bind("<Button-3>", self.show_context_menu)
        self.output_viewer.bind("<Button-2>", self.show_context_menu)
        
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
        
        self.qa_frame = tk.LabelFrame(self.right_panel, text="Client Strategy Q&A with Cerydra", font=self.HEADER_FONT, padx=5, pady=5, bg="white", relief="flat", bd=0)
        self.qa_frame.pack(fill="x", pady=(5, 0))

        tk.Label(self.qa_frame, text="Paste Draft/Text to Evaluate (Optional):", bg="white", font=self.UI_FONT).pack(anchor="w")
        self.draft_text_input = tk.Text(self.qa_frame, height=4, font=self.UI_FONT, wrap="word", relief="solid", borderwidth=1, highlightthickness=2, highlightbackground="white", highlightcolor="black", insertbackground="black")
        self.draft_text_input.pack(fill="x", pady=(0, 10))

        tk.Label(self.qa_frame, text="Your Question (e.g., 'Why was this rejected?'):", bg="white", font=self.UI_FONT).pack(anchor="w", pady=(2, 2))
        self.question_input = tk.Entry(self.qa_frame, font=self.UI_FONT, relief="solid", borderwidth=1, highlightthickness=2, highlightbackground="white", highlightcolor="black")
        self.question_input.pack(fill="x", pady=(0, 10))

        self.query_btn = self._create_clickable_label(self.qa_frame, "🧠 Ask Cerydra", self.run_cerydra_query)
        self.query_btn.pack(anchor="e", pady=5)
        
        self.refresh_file_tree()

        # ==========================================
        #               KEYBOARD SHORTCUTS
        # ==========================================
        self.root.bind("<Command-f>", self._handle_find_shortcut)
        self.root.bind("<Control-f>", self._handle_find_shortcut)

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
    def _create_clickable_label(self, parent_frame, text, command):
        lbl = tk.Label(parent_frame, text=text, font=self.UI_FONT, fg="black", bg="white", cursor="hand2")
        def on_enter(e):
            if str(lbl["state"]) != "disabled": lbl.config(font=self.HOVER_FONT, fg="#666666") 
        def on_leave(e):
            if str(lbl["state"]) != "disabled": lbl.config(font=self.UI_FONT, fg="black") 
        def on_click(e):
            if str(lbl["state"]) != "disabled": command()
        lbl.bind("<Enter>", on_enter)
        lbl.bind("<Leave>", on_leave)
        lbl.bind("<Button-1>", on_click)
        return lbl

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
        
        tk.Button(popup, text="Close", command=popup.destroy, font=self.UI_FONT, bg="#e0e0e0", cursor="hand2").pack(pady=(10, 0))

    # --- LEFT PANEL: LOGIC FUNCTIONS ---
    def _get_target_dir(self, folder_type):
        company = self.company_var.get().strip()
        if not company:
            messagebox.showerror("Error", "Please enter a Company Keyword first!")
            return None
        base_dir = os.path.join("./client_data", company)
        if folder_type == "accepted": return os.path.join(base_dir, "accepted")
        elif folder_type == "feedback": return os.path.join(base_dir, "feedback")
        return base_dir
    
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
        btn_browse = tk.Button(dialog, text="Browse Files...", command=lambda: self._browse_files(folder_type, dialog), font=self.UI_FONT, cursor="hand2")
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

        btn_save = tk.Button(dialog, text="💾 Save Pasted Text", command=save_pasted_text, font=self.UI_FONT, bg="#e0e0e0", cursor="hand2")
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
        model = self.model_var.get()

        if not company:
            messagebox.showwarning("Missing Data", "Company Keyword is required to search source files.")
            return

        self.console.insert(tk.END, f"\n[HYSILENS] Analyzing source for selected snippet using {model}...\n")
        self.console.see(tk.END)
        threading.Thread(target=self._hysilens_thread, args=(selected_text, company, model), daemon=True).start()

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
        
        tk.Button(popup, text="Close", command=popup.destroy, font=self.UI_FONT, bg="#e0e0e0", cursor="hand2").pack(pady=(10, 0))
    
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
        
        tk.Button(popup, text="Close", command=popup.destroy, font=self.UI_FONT, bg="#e0e0e0", cursor="hand2").pack(pady=(10, 0))

    def run_generation(self):
        client_name = self.client_name_var.get().strip()
        company = self.company_var.get().strip()
        action = self.action_var.get()
        model_choice = self.model_var.get() 
        
        if not client_name or not company:
            messagebox.showerror("Error", "Client Name and Company Keyword are required.")
            return
            
        self.run_btn.config(state="disabled", text="Running...")
        self.console.delete(1.0, tk.END) 
        self.output_viewer.delete(1.0, tk.END)
        
        threading.Thread(target=self._process_thread, args=(client_name, company, action, model_choice), daemon=True).start()

    def _process_thread(self, client_name, company, action, model_choice):
        try:
            if action == "post":
                print(f"Ruan Mei: I'm Starting Post Generation for {client_name}...\n")
                self.current_generation_data = generate_iterative_linkedin_posts(client_name, company, model_choice) 
            else:
                print(f"Ruan Mei: I'm Starting Briefing for {client_name} using [{model_choice}]...\n")
                generate_briefing(client_name, company, model_choice)
        except Exception as e: 
            print(f"\nAN ERROR OCCURRED: {e}")
        finally:
            self.run_btn.config(state="normal", text="🚀 RUN AMPHOREUS EXPERIMENT")
            print("\n--- TASK FINISHED ---")
            self.root.after(0, lambda: self.load_output_to_viewer(model_choice))
            self.root.after(0, self.refresh_file_tree)

    def run_rewrite(self):
        company = self.company_var.get().strip()
        client_name = self.client_name_var.get().strip()
        model_choice = self.model_var.get()
        
        # Determine if the user left the instruction blank or with the default text
        style_instruction = self.style_prompt_text.get("1.0", tk.END).strip()
        if "Leave blank for random stylistic noise" in style_instruction:
            style_instruction = ""
            
        if not company or not client_name:
            messagebox.showerror("Error", "Client Name and Company Keyword are required.")
            return

        self.rewrite_btn.config(state="disabled", text="Rewriting...")
        self.console.delete(1.0, tk.END)
        self.output_viewer.delete(1.0, tk.END)
        
        threading.Thread(target=self._rewrite_thread, args=(client_name, company, model_choice, style_instruction), daemon=True).start()

    def _rewrite_thread(self, client_name, company, model_choice, style_instruction):
        try:
            print(f"\n[CYRENE] Starting Iterative Rewrite Pipeline for {client_name}...")
            
            if model_choice == "Gemini 3.1 Pro": filename = f"{company}_gemini_posts.md"
            elif model_choice == "GPT-5": filename = f"{company}_gpt_posts.md"
            elif model_choice == "Claude Opus 4.6": filename = f"{company}_claude_posts.md"
            else: filename = f"{company}_posts.md"
            
            input_filepath = os.path.join("./client_data", company, "output", filename)
            
            if not os.path.exists(input_filepath):
                print(f"Error: Could not find generated drafts at {input_filepath}. Please run generation first.")
                return

            with open(input_filepath, "r", encoding="utf-8") as f:
                raw_draft_text = f.read()

            # --- Extract only the posts, cutting out the preliminary analysis steps ---
            parts = raw_draft_text.split("--- FINAL LINKEDIN POST DRAFTS ---")
            clean_draft_text = parts[-1].strip() if len(parts) > 1 else raw_draft_text.strip()

            cyrene = Cyrene()
            
            if style_instruction:
                print(f"  -> Cyrene rewriting posts iteratively with user instruction: '{style_instruction}'")
            else:
                print(f"  -> Cyrene iteratively applying random stylistic noise...")

            output_filename = f"{company}_rewritten_posts.md"
            output_filepath = os.path.join("./client_data", company, "output", output_filename)
            
            # 1. Initialize the file with headers
            with open(output_filepath, "w", encoding="utf-8") as out_file:
                out_file.write(f"# FINAL REWRITTEN POSTS: {client_name.upper()}\n")
                if style_instruction:
                    out_file.write(f"Style Intent: {style_instruction}\n")
                else:
                    out_file.write(f"Style Intent: Random Stylistic Noise\n")
                out_file.write("="*60 + "\n\n")
            
            # 2. Iterate through Cyrene's generator and append to the file live
            for result in cyrene.rewrite_posts_iteratively(full_draft_text=clean_draft_text, style_instruction=style_instruction):
                idx = result["index"]
                total = result["total"]
                print(f"     Finished rewriting post {idx}/{total}...")
                
                with open(output_filepath, "a", encoding="utf-8") as out_file:
                    out_file.write(f"## POST {idx} REWRITE\n")
                    out_file.write(f"**Step 1: Fact Extraction**\n{result.get('fact_extraction', 'N/A')}\n\n")
                    out_file.write(f"**Step 2: Style Approach**\n{result.get('style_analysis', 'N/A')}\n\n")
                    out_file.write(f"**Step 3: Rewrite Strategy**\n{result.get('strategy', 'N/A')}\n\n")
                    out_file.write(f"**FINAL STYLIZED POST**\n{result.get('final_post', 'N/A')}\n")
                    out_file.write("*" * 50 + "\n\n")
            
            print(f"\nRewrite pipeline complete! Saved to {output_filename}. Loading into viewer...")
            
        except Exception as e:
            print(f"Rewrite Pipeline Error: {e}")
        finally:
            self.rewrite_btn.config(state="normal", text="✨ Rewrite Posts")
            self.root.after(0, lambda: self._load_specific_file(output_filepath))
            self.root.after(0, self.refresh_file_tree)

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

    def load_output_to_viewer(self, model_choice=None):
        if not model_choice:
            model_choice = self.model_var.get()
            
        company = self.company_var.get().strip()
        action = self.action_var.get()
        if not company: 
            return
            
        if action == "brief":
            if model_choice == "Gemini 3.1 Pro": filename = f"{company}_gemini_briefing.md"
            elif model_choice == "GPT-5": filename = f"{company}_gpt_briefing.md"
            elif model_choice == "Claude Opus 4.6": filename = f"{company}_claude_briefing.md"
            else: filename = f"{company}_briefing.md"
        else:
            if model_choice == "Gemini 3.1 Pro": filename = f"{company}_gemini_posts.md"
            elif model_choice == "GPT-5": filename = f"{company}_gpt_posts.md"
            elif model_choice == "Claude Opus 4.6": filename = f"{company}_claude_posts.md"
            else: filename = f"{company}_posts.md"
            
        filepath = os.path.join("./client_data", company, "output", filename)
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

if __name__ == "__main__":
    os.makedirs("./client_data", exist_ok=True)
    os.makedirs("./static/images", exist_ok=True)
    root = tk.Tk()
    app = AmphoreusExperiment(root)
    root.mainloop()