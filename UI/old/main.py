import tkinter as tk
from ui.form_window import launch_form_window_with_callback
from ui.capture_window import launch_capture_window

def main():
    root = tk.Tk()
    root.title("ID Capture App")
    root.geometry("500x300")

    def on_user_info_submitted(user_data):
        curp = user_data.get("curp")
        if not curp:
            return
        # Launch camera capture window
        launch_capture_window(curp, on_capture_complete)

    def on_capture_complete(saved_path):
        if saved_path:
            print(f"[INFO] Image saved at: {saved_path}")
        else:
            print("[INFO] Capture cancelled or failed.")
        # Clear previous widgets (optional cleanup)
        for widget in root.winfo_children():
            widget.destroy()
        # Relaunch form
        launch_form_window_with_callback(root, on_user_info_submitted)

    # Initial launch
    launch_form_window_with_callback(root, on_user_info_submitted)

    root.mainloop()

if __name__ == "__main__":
    main()
