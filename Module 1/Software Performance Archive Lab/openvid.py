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

def open_video(video_file, width, height, x, y, duration):
    process = subprocess.Popen(['ffplay', '-autoexit', '-window_title', f'{video_file}', '-x', str(width), '-y', str(height), '-left', str(x), '-top', str(y), video_file])
    time.sleep(duration)
    process.terminate()

def play_video_thread(video_file, width, height, x, y, duration, root):
    thread = threading.Thread(target=open_video, args=(video_file, width, height, x, y, duration))
    thread.start()
    root.after(duration * 1000, root.quit)  # Schedule root.quit() after the duration in milliseconds
    return thread

def main():
    video_path = "evan-emma-archives/life b-roll"

    root = tk.Tk()
    root.withdraw()

    videos = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(('.mov', '.MOV'))]
    
    if videos:
        # Play a random video at the center of the screen for 3 seconds
        video = random.choice(videos)
        play_video_thread(video, 400, 400, 800, 100, 3, root)
        
    root.mainloop()

if __name__ == "__main__":
	for _ in range(5):
		main()
