#! usr/bin/env python3
from picamera2 import Picamera2, Preview
import time
import numpy as np
import sounddevice as sd
from ultralytics import YOLO
import cv2
import os
from playsound import playsound

DIR_NAME = "output"
BT_FILE = "best.pt"
IMAGE_NAME = "test.jpg"
SOUND_LOCATE = "pisk.wav"

sample_rate = 35000 # 44100 
note_duration = 0.2
fade_duration = 0.1 
melody = [220.00, 293.66, 329.63, 392.00, 440.00, 493.88]

def generate_smooth_1bit_samples(freq):
    t = np.linspace(0, note_duration, int(note_duration * sample_rate), False)
    fade_out = np.exp(-(t - (note_duration - fade_duration)) / fade_duration)
    samples = 32767.0 * np.where(t % (1 / freq * 2) < (1 / freq), 1, -1) * fade_out
    return np.array(samples, dtype=np.int16)
    
def sound_start():
	all_samples = np.concatenate([generate_smooth_1bit_samples(freq) for freq in melody])
	sd.play(all_samples, samplerate=sample_rate)
	sd.wait(10)
		
def check_dir():
	if not os.path.exists(DIR_NAME):
		os.makedirs("output", exist_ok=True)	
	if not os.path.exists(BT_FILE):
		return False
	return True 
    
def main():
	try:
		sound_start()
		if not check_dir():
			print("Not model")
			return
		model = YOLO(BT_FILE)
		picam2 = Picamera2()
		camera_config = picam2.create_preview_configuration()
		picam2.configure(camera_config)
		picam2.start_preview(Preview.QTGL) #QTGL
		picam2.start()
		while True:
			picam2.capture_file(f"{DIR_NAME}/{IMAGE_NAME}")
			img = cv2.imread(f"{DIR_NAME}/{IMAGE_NAME}")
			results = model(img)
			if "keypoints" in results and results["keypoints"] is not None:
                            playsound(SOUND_LOCATE)
			print(results)
	except KeyboardInterrupt:
		pass
	finally:
		sound_start()

if __name__ == "__main__":
	main()
