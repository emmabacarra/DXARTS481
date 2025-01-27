import tkinter as tk

def create_window():
    global window
    window = tk.Tk()
    window.title("Simple Tkinter Window")
    
    button = tk.Button(window, text = "Click Me", command = on_button_click)
    button.pack()
    
    window.mainloop()
    
def on_button_click():
    print("Button was clicked!")
    create_new_button(window, "New Button")
    
def create_new_button(parent, text):
    button = tk.Button(parent, text = text, command = on_button_click)
    button.pack()
    
create_window()