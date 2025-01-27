import pyautogui
import subprocess 
import time
import os


####################### 1 ###########################
##Displaying Pop-up alerts and messages using pyautogui functionalities:
#combined with a fragment of an x-files script....

# Display an alert x2
alert_response = pyautogui.alert(
'In the forest, a large group of loggers are congregated loosely around two people in the center, who are arguing.', ##this is the message displayed
'SCENE 1 OLYMPIC NATIONAL FOREST WASHINGTON STATE; PRESENT DAY') ##this is the "title of the window
print(alert_response)  # Output: 'OK'

alert_response = pyautogui.alert(
'We have to take a chance, one of us has to hike out!', ##this is the message displayed
'That person might not make it in time') ##this is the "title of the window
print(alert_response)  # Output: 'OK'

# Display a confirmation dialog
confirm_response = pyautogui.confirm('Nobody would listen to me')
print(confirm_response)  # Output: 'OK' or 'Cancel'

# Display a prompt dialog
prompt_response = pyautogui.prompt("What about the rest of us? What are we supposed to do? Just wait here until help arrives?")
print(prompt_response)  # Output: user input

# Display a password prompt
password_response = pyautogui.password('We have to take a chance, What is the password?')
print(password_response)  # Output: user input




####################### 2 ###########################

###We can create a new .txt file and open it on a text editor window, then automate text to be written inside it. 
file_path = 'Fine_You_stay_here-Tell_us_how_it_turns_out.txt'; 

open(file_path, 'w').close(); 

##Then we can open the file with TextEditor app in Mac
#Mac
subprocess.run(['open', '-a', 'TextEdit', file_path])
#windows
#subprocess.run(['notepad.exe', file_path])

# Wait a moment to make sure that the text window is active
time.sleep(2)

# Content to write
text_content = '''He pushes past him. The men start to run. Perkins runs with them. Indiscernible yells can be heard.

MAN: Hurry up!

(Perkins runs after Dyer.)

ANOTHER MAN: You're going the wrong way!

(It is sundown, and now nightfall. Animals screech and howl. Dyer trips over a log and grunts. Perkins jumps over the log and kneels down beside him. Dyer checks his ankle.)'''

##alternativley you can copy the contents from a .txt file you already have:
##
text_file = 'evan-emma-archives/Texts/sprawlSickness.txt'
# Read the first 10 lines from the text file 
with open(text_file, 'r') as file: lines = [file.readline() for _ in range(10)] 
# Join the lines into a single string

text_content_file = ''.join(lines)

# Move the TextEdit window to a specific position using AppleScript
subprocess.run(['osascript', '-e', 'tell application "System Events" to set position of first window of process "TextEdit" to {100, 100}'])

# Move the mouse to the calculated position
pyautogui.moveTo(150, 150, duration=1)

# Click the mouse
pyautogui.click() 

# Type the text content from inside Geany
##pyautogui.write(text_content, interval=0.05)

#type the text content from an existing .txt file
pyautogui.write(text_content_file, interval=0.05)
