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

# -------------

def open_random_image():
    image_folder = 'evan-emma-archives/Art'
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


#if __name__ == "__main__":
#	open_random_image()

# -------------

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

#if __name__ == "__main__":
#	for _ in range(5):
#		main()
		
# -------------
		
		
# Initialize pygame mixer
pygame.mixer.init()

# Specify the folder containing sound files
sound_folder = "evan-emma-archives/Music"  # Adjust the path to your folder

# Define the duration in seconds
duration = 5

# List all sound files in the folder
sound_files = [f for f in os.listdir(sound_folder) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac'))]

# Choose a random sound file from the folder
random_sound_file = random.choice(sound_files)
print(random_sound_file)
file_path = os.path.join(sound_folder, random_sound_file)

# Load and play the sound
pygame.mixer.music.load(file_path)
pygame.mixer.music.play()

# Wait for the specified duration
time.sleep(duration)

# Stop the sound
pygame.mixer.music.stop()
print("sound stopped")
# Exit the script
sys.exit()

