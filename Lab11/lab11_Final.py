import cv2
import numpy as np
import time
import threading
import math
from numpy import random
import torch
from torchvision import transforms
from djitellopy import Tello
from pyimagesearch.pid import PID

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import  plot_one_box

WEIGHT = './runs/train/yolov7-lab093/weights/best.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

model = attempt_load(WEIGHT, map_location=device)
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


def check_lens():
    # 棋盤格規格，假設 6x9 的內部角點
    pattern_size = (9, 6)

    # 創建 object points，假設棋盤格在平面上
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # 儲存 object points 和 image points 的列表
    obj_points = []  # 存放 object points
    img_points = []  # 存放 image points

    # drone
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()

    while True:
        # ret, frame = cap.read()
        frame = frame_read.frame

        # if not ret:
            # break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 嘗試尋找棋盤格角點
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)

        if ret:
            # 如果成功找到角點，使用cornerSubPix進一步提高精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 將object points添加到列表
            obj_points.append(objp)

            # 將image points添加到列表
            img_points.append(corners)

            # 在影像上繪製角點
            cv2.drawChessboardCorners(frame, pattern_size, corners, ret)

        cv2.imshow('Calibration', frame)

        print(len(obj_points), len(img_points))
        if len(img_points) > 20:
            print("Collection done")
            break

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    # cap.release()
    cv2.destroyAllWindows()

    # 進行相機校準
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # 將相機內部參數儲存到XML檔
    cv2_file = cv2.FileStorage("camera_params.xml", cv2.FileStorage_WRITE)
    cv2_file.write("camera_matrix", mtx)
    cv2_file.write("dist_coeff", dist)
    cv2_file.release()

#######

def stop(self):
    self.send_rc_control(0, 0, 0, 0)

def keyboard(self, key):
    #global is_flying
    # print("key:", key)
    fb_speed = 50
    lf_speed = 50
    ud_speed = 80
    degree = 10
    if key == ord('1'):
        self.takeoff()
        self.move("up", 50)
        #is_flying = True
    if key == ord('2'):
        self.land()
        #is_flying = False
    if key == ord('3'):
        stop(self)
        print("stop!!!!")
    if key == ord('w'):
        self.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!")
        # time.sleep(10)
    if key == ord('s'):
        self.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!")
    if key == ord('a'):
        self.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!")
    if key == ord('d'):
        self.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!")
    if key == ord('z'):
        self.send_rc_control(0, 0, ud_speed, 0)
        print("down!!!!")
    if key == ord('x'):
        self.send_rc_control(0, 0, (-1) *ud_speed, 0)
        print("up!!!!")
    if key == ord('c'):
        self.send_rc_control(0, 0, 0, degree)
        print("rotate!!!!")
    if key == ord('v'):
        self.send_rc_control(0, 0, 0, (-1) *degree)
        print("counter rotate!!!!")
    if key == ord('5'):
        height = self.get_height()
        print(height)
    if key == ord('6'):
        battery = self.get_battery()
        print(battery)
    time.sleep(0.1)
    stop(self)

#####################################################################

def main():
    # Tello
    drone = Tello()
    drone.connect()
    # drone.set_video_encoder_rate(1)
    # print(drone.get_battery())
    # time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    # vid = cv2.VideoCapture(drone.get_udp_video_address())
    # success, frame = vid.read()
    
    fs = cv2.FileStorage("camera_params.xml", cv2.FILE_STORAGE_READ)
    intri = fs.getNode("camera_matrix").mat()
    distortion = fs.getNode("dist_coeff").mat()

    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    yaw_pid.initialize()
    z_pid.initialize()
    y_pid.initialize()
    
    progress = 0
    subprogress = 0
    haveChanged = False

    label = ""

    direction = "up"
    while True:
        # img
        frame = frame_read.frame

        if subprogress == 0:
            image = letterbox(frame, (640, 640), stride=64, auto=True)[0]

            if device == "cuda":
                image = transforms.ToTensor()(image).to(device).half().unsqueeze(0)
            else:
                image = transforms.ToTensor()(image).to(device).float().unsqueeze(0)

            with torch.no_grad():
                output = model(image)[0]

            output = non_max_suppression_kpt(output, conf_thres=0.25, iou_thres=0.65)[0]

            if output is not None:
                output[:, :4] = scale_coords(image.shape[2:], output[:, :4], frame.shape).round()

                for *xyxy, _, cls in output:
                    label = f'{names[int(cls)]}'
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)

            print (label)
            if label == "cana":
                subprogress = 1
            if label == "melody":
                subprogress = 2

        # Load the predefined dictionary
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

        # Initialize the detector parameters using default values
        parameters = cv2.aruco.DetectorParameters_create()

        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        # print(intri, distortion)
        if len(markerCorners) > 0:
            global rvec, tvec
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intri, distortion)

            for i in range(rvec.shape[0]):
                frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
                frame = cv2.aruco.drawAxis(frame, intri, distortion, rvec[i,:,:], tvec[i,:,:], 10)
                cv2.putText(frame,
                            f"x: {round(tvec[0,0,0], 2)}, y: {round(tvec[0,0,1], 2)}, z: {round(tvec[0,0,2], 2)}",
                            # "x: "+str(round(tvec[0,0,0], 2))+", y: "+str(round(tvec[0,0,1], 2))+", z: "+str(round(tvec[0,0,2], 4))+", deg: "+str(angle_deg)),
                            (0,64),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2,
                            cv2.LINE_AA
                )
                print(f"x: {round(tvec[0,0,0], 2)}, y: {round(tvec[0,0,1], 2)}, z: {round(tvec[0,0,2], 2)}")
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 2:
            x1, y1, w1, h1 = rects[0]
            x2, y2, w2, h2 = rects[1]
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 3)

        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)

        # detect black pixel
        row, col, _ = frame.shape
        THRESHOLD = 30
        frame = cv2.resize(frame, (row // 4, col // 4))
        row, col, _ = frame.shape
        grid = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        for i in range(row):
            for j in range(col):
                pixel = frame[i][j]
                b, g, r = list(map(int, (pixel[0], pixel[1], pixel[2])))
                if b < THRESHOLD and g < THRESHOLD and r < THRESHOLD:
                    grid[i * 3 // row][j * 3 // col] += 1
        print(grid)

        f = False
        if len(markerCorners) > 0:
            for i in range(rvec.shape[0]):
                id = markerIds[i][0]
                print(f"ID: {markerIds}")
                print(f"Progress: {progress}")
                if id == 1 and progress == 0:
                    z = tvec[i, 0, 2]
                    f = False
                    if z > 70:
                        x_update = 0
                        y_update = 0
                        if tvec[i, 0, 0] > 3:
                            x_update = 3
                        elif tvec[i, 0, 0] < -3:
                            x_update = -3
                        if tvec[i,0,1] > -5:
                            y_update = -10
                        elif tvec[i,0,1] < -15:
                            y_update = 10
                        drone.send_rc_control(x_update, 15, y_update, 0)
                        time.sleep(0.1)
                        stop(drone)
                    else:
                        drone.move_up(20)
                        stop(drone)
                        progress = 2
                        f = True
                        break
                if id == 2 and (progress == 1 or progress == 2):
                    progress = 3
                    drone.rotate_clockwise(90)
                    drone.move_right(210)
                    drone.move_back(35)

                if id == 3 and progress == 4:
                    z = tvec[i, 0, 2]
                    if z < 150:
                        x_update = 0
                        y_update = 0
                        if tvec[i, 0, 0] > 15:
                            x_update = 10
                        elif tvec[i, 0, 0] < -15:
                            x_update = -10
                        if tvec[i,0,1] > -5:
                            y_update = -10
                        elif tvec[i,0,1] < -15:
                            y_update = 10

                        R, _ = cv2.Rodrigues(rvec[i])

                        deg = 0
                        if R[2, 0] > 0.1:
                            deg = -25
                        elif R[2, 0] < -0.1:
                            deg = 25

                        drone.send_rc_control(x_update, -20, y_update, deg)
                        time.sleep(0.1)
                        stop(drone)
                    else:
                        drone.land()
                        progress = 5

        if f:
            continue

        print(progress)
        SPEED = 8
        TIME = 0.5
        BLACK_THRESHOLD = 800
        up_grid = grid[0][0] + grid[0][1] + grid[0][2]
        down_grid = grid[2][0] + grid[2][1] + grid[2][2]
        left_grid = grid[0][0] + grid[1][0] + grid[2][0]
        right_grid = grid[0][2] + grid[1][2] + grid[2][2]
        if progress == 1:
            if direction == "none":
                if left_grid > 500:
                    drone.send_rc_control(-SPEED, 0, 0, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "left"
                elif right_grid > 500:
                    drone.send_rc_control(SPEED, 0, 0, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "right"
                else:
                    stop(drone)
                    direction = "none"
                    progress = 2

            elif direction == "upnone":
                left_grid = grid[0][0] + grid[1][0]
                right_grid = grid[0][2] + grid[1][2]
                if left_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(-SPEED, 0, 0, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "left"
                elif right_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(SPEED, 0, 0, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "right"
                else:
                    stop(drone)
                    direction = "none"
                    progress = 2

            elif direction == "downnone":
                left_grid = grid[2][0] + grid[1][0]
                right_grid = grid[2][2] + grid[1][2]
                if left_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(-SPEED, 0, 0, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "left"
                elif right_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(SPEED, 0, 0, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "right"
                else:
                    stop(drone)
                    direction = "none"
                    progress = 2

            elif direction == "right":
                if right_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(SPEED, 0, 0, 0)
                    time.sleep(TIME)
                    stop(drone)
                else:
                    stop(drone)
                    direction = "rightnone"
                    progress = 2
                
            elif direction == "left":
                if grid[0][1] > BLACK_THRESHOLD and subprogress == 2:
                    stop(drone)
                    direction = "up"
                    subprogress = 2
                    progress = 2
                elif left_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(-SPEED, 0, 0, 0)
                    time.sleep(TIME)
                    stop(drone)
                else:
                    stop(drone)
                    direction = "leftnone"
                    progress = 2

        SPEED = 15
        if progress == 2:
            if direction == "none":
                if up_grid > 500:
                    drone.send_rc_control(0, 0, SPEED, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "up"
                elif down_grid > 500:
                    drone.send_rc_control(0, 0, -SPEED, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "down"
                else:
                    stop(drone)
                    direction = "none"
                    progress = 1

            elif direction == "leftnone":
                up_grid = grid[0][0] + grid[0][1]
                down_grid = grid[2][0] + grid[2][1]
                if up_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(0, 0, SPEED, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "up"
                elif down_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(0, 0, -SPEED, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "down"
                else:
                    stop(drone)
                    direction = "none"
                    progress = 1

            elif direction == "rightnone":
                up_grid = grid[0][1] + grid[0][2]
                down_grid = grid[2][1] + grid[2][2]
                if up_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(0, 0, SPEED, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "up"
                elif down_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(0, 0, -SPEED, 0)
                    time.sleep(TIME)
                    stop(drone)
                    direction = "down"
                else:
                    stop(drone)
                    direction = "none"
                    progress = 1

            elif direction == "up":
                if up_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(0, 0, SPEED, 0)
                    time.sleep(TIME)
                    stop(drone)
                else:
                    stop(drone)
                    direction = "upnone"
                    progress = 1

            elif direction == "down":
                if haveChanged == False:
                    if subprogress == 1:
                        subprogress = 2
                    else:
                        subprogress = 1
                    haveChanged = True
                if down_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(0, 0, -SPEED, 0)
                    time.sleep(TIME)
                    stop(drone)
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                    direction = "downnone"
                    progress = 1

        # check face
        elif progress == 3:
            if len(rects) == 2:
                target_x = (x1 + w1 + x2) / 2
                print(f"IU1: {x1 + w1}\nIU2: {x2}")
                if target_x > 480 + 10:
                    print(f"Deviated: {target_x}")
                    drone.send_rc_control(15, 0, 0, 0)
                    time.sleep(0.1)
                elif target_x < 480 - 10:
                    print(f"Deviated: {target_x}")
                    drone.send_rc_control(-15, 0, 0, 0)
                    time.sleep(0.1)
                else:
                    print("Two IU detected, go")
                    drone.move_forward(200)
                    drone.move_left(30)
                    drone.rotate_clockwise(180)
                    progress = 4
                stop(drone)
            else:
                continue
        # control
        if key != -1:
        #     if "tvec" in globals():
        #         print(f"x: {round(tvec[0,0,0], 2)}, y: {round(tvec[0,0,1], 2)}, z: {round(tvec[0,0,2], 2)}, deg: {R[2, 0]}")
            keyboard(drone, key)
        #     if "tvec" in globals():
        #         print(f"x: {round(tvec[0,0,0], 2)}, y: {round(tvec[0,0,1], 2)}, z: {round(tvec[0,0,2], 2)}, deg: {R[2, 0]}")

        time.sleep(0.01)
    
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    #check_lens()
    main()


