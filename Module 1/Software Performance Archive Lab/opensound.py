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
