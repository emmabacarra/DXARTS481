import subprocess
import time
import functions as f
import os

    
if __name__ == "__main__":
    subprocess.Popen(['python', 'openalbumcover.py'])
    subprocess.Popen(['python', 'opensong.py'])

    for _ in range(4):
        subprocess.Popen(['python', 'openvideo.py'])
        time.sleep(2)
        
    for _ in range(10):
        subprocess.Popen(['python', 'opentextures.py'])
        time.sleep(0.5)
        time.sleep(2)
    
    print('Finished running programs.')
