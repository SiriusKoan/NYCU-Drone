import cv2
import numpy as np
import time
import threading
import math
from djitellopy import Tello
from pyimagesearch.pid import PID


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

    direction = "right"
    while True:
        # img
        frame = frame_read.frame

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

            if progress == 2:
                drone.land()

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

        if len(markerCorners) > 0:
            for i in range(rvec.shape[0]):
                id = markerIds[i][0]
                print(f"ID: {markerIds}")
                print(f"Progress: {progress}")
                if id == 4 and progress == 0:
                    z = tvec[i, 0, 2]
                    if z > 60:
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
                        drone.send_rc_control(x_update, 10, y_update, 0)
                        time.sleep(0.5)
                        stop(drone)
                    else:
                        stop(drone)
                        progress = 1
            drone.send_rc_control(20, 0, 0, 0)
            time.sleep(0.5)

        print(progress)
        SPEED = 8
        BLACK_THRESHOLD = 100
        up_grid = grid[0][0] + grid[0][1] + grid[0][2]
        down_grid = grid[2][0] + grid[2][1] + grid[2][2]
        left_grid = grid[0][0] + grid[1][0] + grid[2][0]
        right_grid = grid[0][2] + grid[1][2] + grid[2][2]
        if progress == 1:
            if direction == "none":
                if right_grid > 500:
                    drone.send_rc_control(SPEED, 0, 0, 0)
                    time.sleep(0.5)
                    stop(drone)
                    direction = "right"
                elif left_grid > 500:
                    drone.send_rc_control(-SPEED, 0, 0, 0)
                    time.sleep(0.5)
                    stop(drone)
                    direction = "left"
                else:
                    stop(drone)
                    direction = "none"
                    progress = 2

            elif direction == "right":
                if right_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(SPEED, 0, 0, 0)
                    time.sleep(0.5)
                    stop(drone)
                # elif grid[0][2] > 1000:
                else:
                    stop(drone)
                    direction = "none"
                    progress = 2
                
            elif direction == "left":
                if left_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(-SPEED, 0, 0, 0)
                    time.sleep(0.5)
                    stop(drone)
                else:
                    stop(drone)
                    direction = "none"
                    progress = 2

        SPEED = 15
        if progress == 2:
            if direction == "none":
                if up_grid > 500:
                    drone.send_rc_control(0, 0, SPEED, 0)
                    time.sleep(0.5)
                    stop(drone)
                    direction = "up"
                elif down_grid > 500:
                    drone.send_rc_control(0, 0, -SPEED, 0)
                    time.sleep(0.5)
                    stop(drone)
                    direction = "down"
                else:
                    stop(drone)
                    direction = "none"
                    progress = 1

            elif direction == "up":
                if up_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(0, 0, SPEED, 0)
                    time.sleep(0.5)
                    stop(drone)
                else:
                    stop(drone)
                    direction = "none"
                    progress = 1

            elif direction == "down":
                if down_grid > BLACK_THRESHOLD:
                    drone.send_rc_control(0, 0, -SPEED, 0)
                    time.sleep(0.5)
                    stop(drone)
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                    direction = "none"
                    progress = 1

        # cv2.putText(
        #     frame,
        #     f"Progress: {progress}",
        #     (0,64),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0,255,0),
        #     2,
        #     cv2.LINE_AA
        # )
        #         if id == 0 and progress == 3:
        #             MAX_SPEED_THRESHOLD = 25
        #             z_update = tvec[i, 0, 2] - 75
        #             print("org_z: " + str(z_update))
        #             z_update = z_pid.update(z_update, sleep=0)
        #             print("pid_z: " + str(z_update))
        #             if z_update > MAX_SPEED_THRESHOLD:
        #                 z_update = MAX_SPEED_THRESHOLD
        #             elif z_update < -MAX_SPEED_THRESHOLD:
        #                 z_update = -MAX_SPEED_THRESHOLD

        #             x_update = 0
        #             y_update = 0
        #             if tvec[i, 0, 0] > 15:
        #                 x_update = 10
        #             elif tvec[i, 0, 0] < -15:
        #                 x_update = -10
        #             if tvec[i,0,1] > -5:
        #                 y_update = -10
        #             elif tvec[i,0,1] < -15:
        #                 y_update = 10

        #             R, _ = cv2.Rodrigues(rvec[i])

        #             deg = 0
        #             if R[2, 0] > 0.1:
        #                 deg = -25
        #             elif R[2, 0] < -0.1:
        #                 deg = 25

        #             drone.send_rc_control(x_update, int(z_update), y_update, deg)
        #             time.sleep(0.01)

        #         if [3] in markerIds and [0] not in markerIds:
        #             progress = 4

        #         if id == 3 and progress == 4:
        #             INTERVAL = 1
        #             z = tvec[i, 0, 2]
        #             if z > 100:
        #                 x_update = 0
        #                 y_update = 0
        #                 if tvec[i, 0, 0] > 15:
        #                     x_update = 10
        #                 elif tvec[i, 0, 0] < -15:
        #                     x_update = -10
        #                 if tvec[i,0,1] > -5:
        #                     y_update = -10
        #                 elif tvec[i,0,1] < -15:
        #                     y_update = 10

        #                 R, _ = cv2.Rodrigues(rvec[i])

        #                 deg = 0
        #                 if R[2, 0] > 0.1:
        #                     deg = -25
        #                 elif R[2, 0] < -0.1:
        #                     deg = 25

        #                 drone.send_rc_control(x_update, 20, y_update, deg)
        #                 time.sleep(0.5)
        #                 stop(drone)
        #             else:
        #                 drone.rotate_clockwise(90)
        #                 # stop(drone)
        #                 progress = 5

        #         if id == 4 and progress == 5:
        #             INTERVAL = 1
        #             z = tvec[i, 0, 2]
        #             if z > 85:
        #                 x_update = 0
        #                 y_update = 0
        #                 if tvec[i, 0, 0] > 15:
        #                     x_update = 10
        #                 elif tvec[i, 0, 0] < -15:
        #                     x_update = -10
        #                 if tvec[i,0,1] > -5:
        #                     y_update = -10
        #                 elif tvec[i,0,1] < -15:
        #                     y_update = 10

        #                 R, _ = cv2.Rodrigues(rvec[i])

        #                 deg = 0
        #                 if R[2, 0] > 0.1:
        #                     deg = -25
        #                 elif R[2, 0] < -0.1:
        #                     deg = 25

        #                 drone.send_rc_control(x_update, 20, y_update, deg)
        #                 time.sleep(0.5)
        #                 stop(drone)
        #             else:
        #                 progress = 6
        #                 drone.move("left", 250)

        #         if id == 5 and progress == 6:
        #             z = tvec[i, 0, 2]
        #             if z < 150:
        #                 x_update = 0
        #                 y_update = 0
        #                 if tvec[i, 0, 0] > 15:
        #                     x_update = 10
        #                 elif tvec[i, 0, 0] < -15:
        #                     x_update = -10
        #                 if tvec[i,0,1] > -5:
        #                     y_update = -10
        #                 elif tvec[i,0,1] < -15:
        #                     y_update = 10

        #                 R, _ = cv2.Rodrigues(rvec[i])

        #                 deg = 0
        #                 if R[2, 0] > 0.1:
        #                     deg = -25
        #                 elif R[2, 0] < -0.1:
        #                     deg = 25

        #                 drone.send_rc_control(x_update, -20, y_update, deg)
        #                 time.sleep(0.5)
        #                 stop(drone)
        #             else:
        #                 drone.land()
        #                 progress = 7
        # else:
        #     if progress == 6:
        #         drone.send_rc_control(0, -20, 0, 0)
        #         time.sleep(0.5)
        #         stop(drone)

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


