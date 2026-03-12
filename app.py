import os
import shutil
import threading
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

# Import your existing worker functions!
from post import generate_iterative_linkedin_posts
from brief import generate_ghostwriter_briefing

class PrintLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
    def flush(self):
        pass

class RuanMeiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ruan Mei")
        self.root.geometry("1300x800") 
        self.root.configure(padx=10, pady=10, bg="white")

        # --- BUTTON STYLE CONFIG ---
        # We define these once to keep the code clean
        self.btn_style = {
            "bg": "white",
            "fg": "black",
            "relief": "solid",
            "borderwidth": 1,
            "activebackground": "#f0f0f0", # Slight gray when clicked
            "cursor": "hand2"
        }

        self.chibi_images = []
        self.image_refs = []
        static_dir = "./static/images"
        
        if os.path.exists(static_dir):
            for file in os.listdir(static_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    self.chibi_images.append(os.path.join(static_dir, file))

        # --- LAYOUT ---
        self.left_panel = tk.Frame(root, bg="white")
        self.left_panel.pack(side="left", fill="y", expand=False, padx=(0, 10))
        
        self.right_panel = tk.Frame(root, bg="white")
        self.right_panel.pack(side="right", fill="both", expand=True)

        # --- 1. CLIENT DETAILS ---
        frame_details = tk.LabelFrame(self.left_panel, text="1. Client Details", padx=10, pady=10, bg="white")
        frame_details.pack(fill="x", pady=5)

        tk.Label(frame_details, text="Client Name:", bg="white").grid(row=0, column=0, sticky="w")
        self.client_name_var = tk.StringVar()
        tk.Entry(frame_details, textvariable=self.client_name_var, width=30, relief="solid", borderwidth=1).grid(row=0, column=1, padx=10)

        tk.Label(frame_details, text="Company Keyword:", bg="white").grid(row=1, column=0, sticky="w", pady=5)
        self.company_var = tk.StringVar()
        tk.Entry(frame_details, textvariable=self.company_var, width=30, relief="solid", borderwidth=1).grid(row=1, column=1, padx=10)

        # Voice Line 1
        v1_frame = tk.Frame(self.left_panel, bg="white")
        v1_frame.pack(fill="x", pady=2)
        self._place_chibi(v1_frame, side="left", size=(70, 70))
        tk.Label(v1_frame, text="\"A picture of grace and elegance.\"", font=("Georgia", 10, "italic"), fg="#888888", bg="white").pack(side="left", padx=10)

        # --- 2. FILE MANAGEMENT ---
        frame_files = tk.LabelFrame(self.left_panel, text="2. Upload Documents", padx=10, pady=10, bg="white")
        frame_files.pack(fill="x", pady=5)
        
        tk.Label(frame_files, text="Make sure Company Keyword is filled out first.", fg="gray", bg="white").pack(pady=(0, 5))

        tk.Button(frame_files, text="Add Base Files (Transcripts/Profile)", command=lambda: self.add_files("base"), width=35, **self.btn_style).pack(pady=2)
        tk.Button(frame_files, text="Add Accepted Posts (Optional)", command=lambda: self.add_files("accepted"), width=35, **self.btn_style).pack(pady=2)
        tk.Button(frame_files, text="Add Rejected Posts (Optional)", command=lambda: self.add_files("blocked"), width=35, **self.btn_style).pack(pady=2)

        # Voice Line 2
        v2_frame = tk.Frame(self.left_panel, bg="white")
        v2_frame.pack(fill="x", pady=2)
        self._place_chibi(v2_frame, side="right", size=(70, 70))
        tk.Label(v2_frame, text="\"What do we have here?\"", font=("Georgia", 10, "italic"), fg="#888888", bg="white").pack(side="right", padx=10)

        # --- 3. GENERATION ---
        frame_gen = tk.LabelFrame(self.left_panel, text="3. Generation", padx=10, pady=10, bg="white")
        frame_gen.pack(fill="x", pady=5)

        self.action_var = tk.StringVar(value="brief")
        tk.Radiobutton(frame_gen, text="Generate Briefing?", variable=self.action_var, value="brief", bg="white", activebackground="white").pack(anchor="w")
        tk.Radiobutton(frame_gen, text="Generate 10 LinkedIn Posts?", variable=self.action_var, value="post", bg="white", activebackground="white").pack(anchor="w")

        # The Main Run Button
        self.run_btn = tk.Button(frame_gen, text="🚀 PLEASE, RUAN MEI!", command=self.run_generation, font=("Arial", 12, "bold"), pady=8, **self.btn_style)
        self.run_btn.pack(fill="x", pady=10)

        # Voice Line 3
        v3_frame = tk.Frame(self.left_panel, bg="white")
        v3_frame.pack(fill="x", pady=2)
        self._place_chibi(v3_frame, side="left", size=(70, 70))
        tk.Label(v3_frame, text="\"You hold some academic value, I suppose...\"", font=("Georgia", 10, "italic"), fg="#888888", bg="white").pack(side="left", padx=10)

        # --- CONSOLE OUTPUT ---
        tk.Label(self.left_panel, text="Console Output:", bg="white").pack(anchor="w", pady=(5, 0))
        self.console = scrolledtext.ScrolledText(self.left_panel, width=55, height=12, bg="#1e1e1e", fg="#00ff00", font=("Courier", 10), borderwidth=1, relief="solid")
        self.console.pack(fill="both", expand=True)

        sys.stdout = PrintLogger(self.console)

        # --- RIGHT PANEL: OUTPUT VIEWER ---
        header_frame = tk.Frame(self.right_panel, bg="white")
        header_frame.pack(fill="x", pady=(0, 10))
        
        self._place_chibi(header_frame, side="left", size=(50, 50))
        tk.Label(header_frame, text="Document Viewer", font=("Arial", 14, "bold"), bg="white").pack(side="left", padx=(5, 0))
        
        # Manual Load Button with style
        tk.Button(header_frame, text="🔄 Manually Load Output", command=self.load_output_to_viewer, **self.btn_style, padx=10).pack(side="right")

        self.output_viewer = scrolledtext.ScrolledText(self.right_panel, bg="#ffffff", fg="#333333", font=("Helvetica", 11), wrap="word", highlightthickness=1, highlightbackground="black", relief="flat")
        self.output_viewer.pack(fill="both", expand=True)

    def _place_chibi(self, parent_frame, side="left", size=(60, 60)):
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
        except Exception as e:
            print(f"Error loading chibi: {e}")

    # --- LOGIC FUNCTIONS ---
    def _get_target_dir(self, folder_type):
        company = self.company_var.get().strip()
        if not company:
            messagebox.showerror("Error", "Please enter a Company Keyword first!")
            return None
        base_dir = os.path.join("./client_data", company)
        if folder_type == "accepted": return os.path.join(base_dir, "accepted")
        elif folder_type == "blocked": return os.path.join(base_dir, "blocked")
        return base_dir

    def add_files(self, folder_type):
        target_dir = self._get_target_dir(folder_type)
        if not target_dir: return
        files = filedialog.askopenfilenames(title=f"Select files for {folder_type}")
        if files:
            os.makedirs(target_dir, exist_ok=True)
            for file in files:
                try:
                    shutil.copy(file, target_dir)
                    print(f"Added {os.path.basename(file)} to {folder_type} folder.")
                except Exception as e: print(f"Failed to copy {file}: {e}")

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
        threading.Thread(target=self._process_thread, args=(client_name, company, action), daemon=True).start()

    def _process_thread(self, client_name, company, action):
        try:
            if action == "post":
                print(f"Ruan Mei: I'm Starting Post Generation for {client_name}...\n")
                generate_iterative_linkedin_posts(client_name, company)
            else:
                print(f"Ruan Mei: I'm Starting Briefing for {client_name}...\n")
                generate_ghostwriter_briefing(client_name, company)
        except Exception as e: print(f"\nAN ERROR OCCURRED: {e}")
        finally:
            self.run_btn.config(state="normal", text="🚀 PLEASE, RUAN MEI!")
            print("\n--- TASK FINISHED ---")
            self.root.after(0, self.load_output_to_viewer)

    def load_output_to_viewer(self):
        company = self.company_var.get().strip()
        action = self.action_var.get()
        if not company: return
        if action == "brief":
            filepath = os.path.join("./client_data", company, f"{company}_ruanmei_briefing.md")
        else:
            filepath = os.path.join("./client_data", company, f"{company}_ruanmei_posts.txt")
        self.output_viewer.delete(1.0, tk.END)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                self.output_viewer.insert(tk.END, content)
                print(f"Loaded {os.path.basename(filepath)} into viewer.")
            except Exception as e: self.output_viewer.insert(tk.END, f"Error reading file:\n{e}")
        else: self.output_viewer.insert(tk.END, "Waiting for output...")

if __name__ == "__main__":
    os.makedirs("./client_data", exist_ok=True)
    os.makedirs("./static/images", exist_ok=True)
    root = tk.Tk()
    app = RuanMeiGUI(root)
    root.mainloop()