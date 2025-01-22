import subprocess
import time
from openimage import open_random_image

# Paths to the Python programs
#program1 = 'cookbook.py'
program1 = 'opentext.py'
program2 = 'openimage.py'
program2_1 = 'openimage2.py'
program3 = 'openvid.py'
program4 = 'opensound.py'

# Running the programs
if __name__ == "__main__":
	
	for _ in range(4):
		subprocess.Popen(['python', program1])
		time.sleep(2)
		
	for _ in range(10):
		subprocess.Popen(['python', program2])
		time.sleep(0.5)
		subprocess.Popen(['python', program2_1])
		time.sleep(2)
		
	for _ in range(8):
		p1 = subprocess.Popen(['python', program3])
		time.sleep(2)
		if _ < 3:
			p2 = subprocess.Popen(['python', program4])
		time.sleep(2)
		
	print('Finished running programs.')
