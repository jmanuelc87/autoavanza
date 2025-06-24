import tkinter as tk

def launch_form_window_with_callback(root, on_continue):
    """
    Display a form to enter First Name, Second Name, Last Name, CURP.
    Calls on_continue(data) with a dict when 'Continue' button is clicked.
    """
    # Clear root window if needed
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Enter your details", font=("Arial", 16)).pack(pady=10)

    labels = ["First Name", "Second Name", "Last Name", "CURP"]
    entries = {}

    for label_text in labels:
        frame = tk.Frame(root)
        frame.pack(pady=5, fill='x', padx=20)
        label = tk.Label(frame, text=label_text, width=12, anchor='w')
        label.pack(side='left')
        entry = tk.Entry(frame)
        entry.pack(side='left', fill='x', expand=True)
        entries[label_text.lower().replace(" ", "_")] = entry

    def on_click():
        data = {key: entry.get().strip() for key, entry in entries.items()}
        if not data['curp']:
            tk.messagebox.showerror("Error", "CURP is required")
            return
        on_continue(data)

    btn = tk.Button(root, text="Continue to Capture", command=on_click)
    btn.pack(pady=20)
