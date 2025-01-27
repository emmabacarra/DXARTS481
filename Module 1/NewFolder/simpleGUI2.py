import tkinter as tk
import random

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
    
def create_new_button(parent, text, command):
    button = tk.Button(parent, text = text, command = command)
    button.pack()
    
create_window()