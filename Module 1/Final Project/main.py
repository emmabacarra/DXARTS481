import subprocess
import time
import functions as f


if __name__ == "__main__":
    subprocess.Popen(['python', f.open_image_until_exit('archive/kendrick_lamar_album.png')])
    subprocess.Popen(['python', f.open_media('archive/music/luther.mp4', duration=30)])
    
    for _ in range(4):
        subprocess.Popen(['python', f.open_random_video('archive/life_b-roll')])
        time.sleep(2)
        
    for _ in range(10):
        subprocess.Popen(['python', f.open_random_image('archive/textures')])
        time.sleep(0.5)
        subprocess.Popen(['python', f.open_random_image('archive/tiles')])
        time.sleep(2)
		
	# for _ in range(8):
	# 	p1 = subprocess.Popen(['python', program3])
	# 	time.sleep(2)
	# 	if _ < 3:
	# 		p2 = subprocess.Popen(['python', program4])
	# 	time.sleep(2)

    print('Finished running programs.')
