import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import cv2
from logic.image_utils import detect_and_warp_id, warp_perspective_with_points, detect_corners

def launch_preview_window(image, curp, on_complete):
    preview_window = tk.Toplevel()
    preview_window.title("Preview Captured Image")

    # Convert BGR to RGB and create PIL image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Detect initial corners (returns 4 points [(x,y), ...])
    corners = detect_corners(image)
    if corners is None:
        messagebox.showerror("Error", "Could not detect ID corners automatically.")
        preview_window.destroy()
        return

    # Canvas for showing image and drawing polygon + points
    canvas = tk.Canvas(preview_window, width=pil_img.width, height=pil_img.height)
    canvas.pack()

    imgtk = ImageTk.PhotoImage(pil_img)
    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    points = corners[:]  # mutable copy of corners
    point_handles = []
    polygon_handle = None
    selected_point_index = None
    radius = 8

    def draw_polygon():
        nonlocal polygon_handle, point_handles
        # Remove old drawings
        if polygon_handle:
            canvas.delete(polygon_handle)
        for h in point_handles:
            canvas.delete(h)
        # Draw filled semi-transparent polygon (use stipple for transparency)
        polygon_handle = canvas.create_polygon(
            [coord for point in points for coord in point],
            outline='lime', fill='lime', width=2, stipple='gray25'
        )
        # Draw corner points as circles
        point_handles.clear()
        for (x, y) in points:
            h = canvas.create_oval(
                x-radius, y-radius, x+radius, y+radius,
                fill='yellow', outline='black', width=2
            )
            point_handles.append(h)

    def find_point_index(x, y):
        # Return index of point close to (x,y) or None
        for i, (px, py) in enumerate(points):
            if abs(px - x) <= radius and abs(py - y) <= radius:
                return i
        return None

    def on_mouse_down(event):
        nonlocal selected_point_index
        idx = find_point_index(event.x, event.y)
        if idx is not None:
            selected_point_index = idx

    def on_mouse_move(event):
        nonlocal selected_point_index
        if selected_point_index is not None:
            # Clamp points inside canvas bounds
            x = max(0, min(event.x, pil_img.width))
            y = max(0, min(event.y, pil_img.height))
            points[selected_point_index] = (x, y)
            draw_polygon()

    def on_mouse_up(event):
        nonlocal selected_point_index
        selected_point_index = None

    # Bind mouse events to canvas
    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    draw_polygon()

    def on_confirm_and_align():
        # Use user-edited points for warp
        aligned_img = warp_perspective_with_points(image, points)
        if aligned_img is None:
            messagebox.showerror("Error", "Warped image too small or invalid. Check points.")
            return
        preview_window.destroy()
        show_aligned_window(aligned_img, curp, on_complete)

    confirm_btn = ttk.Button(preview_window, text="✅ Confirm and Align", command=on_confirm_and_align)
    confirm_btn.pack(pady=10)

def show_aligned_window(aligned_img, curp, on_complete):
    aligned_window = tk.Toplevel()
    aligned_window.title("Aligned ID Preview")

    img_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(img_pil)

    label = ttk.Label(aligned_window, image=imgtk)
    label.image = imgtk
    label.pack(pady=10)

    def save_image():
        os.makedirs("captures/INE", exist_ok=True)
        save_path = f"captures/INE/{curp}_INE.jpg"
        cv2.imwrite(save_path, aligned_img)
        aligned_window.destroy()
        on_complete(save_path)

    save_btn = ttk.Button(aligned_window, text="💾 Save Image", command=save_image)
    save_btn.pack(pady=10)
