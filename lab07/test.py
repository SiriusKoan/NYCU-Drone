import cv2
import time
from djitellopy import Tello

tello = Tello()
tello.connect(False)

# tello.send_command_with_return("port 8890 11111")

tello.streamon()
while True:
	img=tello.get_frame_read()
	cv2.imshow("Image",img)
	cv2.waitKey(1)