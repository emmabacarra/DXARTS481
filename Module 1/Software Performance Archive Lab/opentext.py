import pyautogui
import subprocess 
import time
import os
import random

text_folder = 'evan-emma-archives/Texts'
text_files = [f for f in os.listdir(text_folder)]

dict = {}

for text_file in text_files:
	with open(os.path.join(text_folder, text_file), 'r') as text:
		dict[text_file] = text.read()
		
#print(dict)

random_file = random.choice(text_files)

alert_response = pyautogui.alert(dict[random_file])
