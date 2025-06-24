import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import numpy as np
from ui.preview_window import launch_preview_window

# Constants
SCAN_WIDTH = 500
SCAN_HEIGHT = 300

def launch_capture_window(curp, on_capture_complete):
    window = tk.Toplevel()
    window.title("Capture INE ID")
    window.geometry("700x600")
    window.resizable(False, False)

    video_label = ttk.Label(window)
    video_label.pack(pady=10)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    stop_event = threading.Event()

    # Define scan area (centered)
    def draw_overlay(frame):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        x1 = (w - SCAN_WIDTH) // 2
        y1 = (h - SCAN_HEIGHT) // 2
        x2 = x1 + SCAN_WIDTH
        y2 = y1 + SCAN_HEIGHT

        # Draw semi-transparent dark overlay
        dark = np.zeros_like(frame, dtype=np.uint8)
        dark[:] = (0, 0, 0)
        alpha = 0.6
        blended = cv2.addWeighted(frame, 1 - alpha, dark, alpha, 0)

        # Clear the scan area (make it transparent)
        blended[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        # Draw scan area border
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return blended, (x1, y1, x2, y2)

    def update_frame():
        if stop_event.is_set():
            return

        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        overlayed, scan_rect = draw_overlay(frame)
        rgb_frame = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        window.after(10, update_frame)

    def capture_image():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture image")
            return

        frame = cv2.flip(frame, 1)
        _, (x1, y1, x2, y2) = draw_overlay(frame)

        # Crop the scan area from the captured image
        cropped = frame[y1:y2, x1:x2]

        stop_event.set()
        cap.release()
        window.destroy()

        # Send to preview window
        launch_preview_window(cropped, curp, on_capture_complete)

    capture_button = ttk.Button(window, text="📸 Capture Image", command=capture_image)
    capture_button.pack(pady=10)

    update_frame()