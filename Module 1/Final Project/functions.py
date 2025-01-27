import random
import os
import tkinter as tk
import PIL
from PIL import Image, ImageTk
import cv2
import pygame
import time
import sys
import threading

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




def open_random_video(video_folder):
    width = 400  # Width of the video window
    height = 300  # Height of the video window

    # Initialize the main root window
    root = tk.Tk()
    root.withdraw()  # Hide the main root window

    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Randomize window position
    x = random.randint(screen_width // 2, screen_width - width)
    y = random.randint(0, screen_height - height)

    # Get a list of video files
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print("No video files found in the folder.")
        return

    # Choose a random video
    random_video = random.choice(video_files)
    video_path = os.path.join(video_folder, random_video)

    # Open the video window
    video_window = tk.Toplevel(root)
    video_window.geometry(f"{width}x{height}+{x}+{y}")
    video_window.title("Random Video Player")

    # Create a Canvas to display video frames
    canvas = tk.Canvas(video_window, width=width, height=height)
    canvas.pack()

    # Function to play video frames
    def play_video():
        cap = cv2.VideoCapture(video_path)

        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (width, height))  # Resize frame to match the window
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color to RGB for Tkinter
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                canvas.create_image(0, 0, anchor=tk.NW, image=img)
                canvas.image = img
                video_window.after(10, update_frame)  # Update frame every 10 ms
            else:
                cap.release()
                video_window.destroy()  # Close the video window when the video ends

        update_frame()

    play_video()
    root.mainloop()


def open_random_sound(sound_folder, duration=5):
    # Initialize pygame mixer
    pygame.mixer.init()

    # List all sound files in the folder
    sound_files = [f for f in os.listdir(sound_folder) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac'))]

    # Choose a random sound file from the folder
    random_sound_file = random.choice(sound_files)
    file_path = os.path.join(sound_folder, random_sound_file)

    # Load and play the sound
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    # Wait for the specified duration
    time.sleep(duration)

    # Stop the sound
    pygame.mixer.music.stop()

    # Exit the script
    sys.exit()


def open_media(file_path, duration=5):
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension in ['.mp3']:
        # Play audio using pygame
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for the specified duration
        time.sleep(duration)

        # Stop the audio
        pygame.mixer.music.stop()

    elif file_extension in ['.mp4']:
        # Play video using OpenCV
        cap = cv2.VideoCapture(file_path)
        start_time = time.time()

        if not cap.isOpened():
            print(f"Error: Unable to open video file {file_path}")
            return

        # Play video for the specified duration
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Video Playback", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print(f"Unsupported file type: {file_extension}")
        return

    sys.exit()


def open_image_until_exit(image_path, width=400, height=400):
    def display_image():
        # Create the root Tkinter window
        root = tk.Tk()
        root.title("Image Viewer")
        root.geometry(f"{width}x{height}")
        root.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable manual closing

        # Open and display the image
        img = Image.open(image_path)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        panel = tk.Label(root, image=img_tk)
        panel.image = img_tk
        panel.pack()

        root.mainloop()

    # Run the Tkinter window in a separate thread
    thread = threading.Thread(target=display_image, daemon=True)
    thread.start()
