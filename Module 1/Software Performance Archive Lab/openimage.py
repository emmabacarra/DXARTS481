import os
import random
import subprocess
import threading
import tkinter as tk
import PIL
from PIL import Image, ImageTk
import time
import sys
import pygame

def open_random_image(image_folder):
    width = 200
    height = 200
    display_time = 3000  # time in milliseconds (e.g., 5000ms = 5 seconds)

    root = tk.Tk()
    root.withdraw()
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = random.randint(screen_width // 2, screen_width - width)
    y = random.randint(0, screen_height - height)

    image_files = [f for f in os.listdir(image_folder)] # if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    if image_files:
        random_image = random.choice(image_files)
        image_path = os.path.join(image_folder, random_image)

        image_window = tk.Toplevel(root)
        image_window.geometry(f"{width}x{height}+{x}+{y}")

        img = Image.open(image_path)
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(image_window, image=img)
        panel.image = img
        panel.pack()

        root.after(display_time, root.quit)  # Close the root window after `display_time` milliseconds
        root.mainloop()


if __name__ == "__main__":
	open_random_image('evan-emma-archives/Art')
