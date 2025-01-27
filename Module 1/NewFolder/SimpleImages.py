import tkinter as tk
from PIL import Image, ImageTk
import random
import os

def load_button_names(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

button_names = load_button_names('tiny_shakespeare.txt')

def create_window():
    global window
    window = tk.Tk()
    window.title("Create Buttons")
    
    create_new_button(window, 'Shakespearify', button_function)
    
    window.mainloop()
    
def button_function():
    print('Shakespeare!')
    new_button_text = random.choice(button_names)
    create_new_button(window, new_button_text, button_function)
    open_random_image('images')
    
def create_new_button(parent, text, command):
    button = tk.Button(parent, text = text, command = command)
    button.pack()
    
def open_random_image(folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if image_files:
        random_image = random.choice(image_files)
        image_path = os.path.join(folder, random_image)
        
        image_window = tk.Toplevel()
        image_window.title(random_image)
        
        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(image_window, image = img)
        panel.image = img
        panel.pack()
        
create_window()
    