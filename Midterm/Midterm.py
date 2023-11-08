import cv2
import numpy as np
import time
import threading
import math
from djitellopy import Tello
from pyimagesearch.pid import PID


class Drone:
    def __init__(self, interval=1):
        self.drone = Tello()
        self.progress = 0
        self.interval = interval

    def connect(self):
        self.drone.connect()
        self.drone.streamon()

    def stop(self):
        self.drone.send_rc_control(0, 0, 0, 0)

    def takeoff(self):
        self.drone.takeoff()

    def land(self):
        self.drone.land()

    def sleep(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            time.sleep(self.interval)
        return wrapper

    @sleep
    def move(self, x, y, z, deg):
        self.drone.send_rc_control(x, y, z, deg)

    @sleep
    def move_forward(self, speed):
        self.drone.send_rc_control(0, speed, 0, 0)

    @sleep
    def move_backward(self, speed):
        self.drone.send_rc_control(0, -speed, 0, 0)

    @sleep
    def move_left(self, speed):
        self.drone.send_rc_control(-speed, 0, 0, 0)

    @sleep
    def move_right(self, speed):
        self.drone.send_rc_control(speed, 0, 0, 0)

    @sleep
    def move_up(self, speed):
        self.drone.send_rc_control(0, 0, speed, 0)

    @sleep
    def move_down(self, speed):
        self.drone.send_rc_control(0, 0, -speed, 0)

    def adjust(self):
        self.frame_read = drone.get_frame_read()

        fs = cv2.FileStorage("camera_params.xml", cv2.FILE_STORAGE_READ)
        intri = fs.getNode("camera_matrix").mat()
        distortion = fs.getNode("dist_coeff").mat()

        self.z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

        self.yaw_pid.initialize()
        self.z_pid.initialize()
        self.y_pid.initialize()

    def get_frame(self):
        return self.frame_read.frame

    def get_cur_frame(self):
        return self.frame

    def draw_detected_markers(self):
        try:
            self.frame = cv2.aruco.drawDetectedMarkers(
                self.frame, markerCorners, markerIds
            )
        except Exception as e:
            print(e)
        return self.frame

    def draw_axis(self, intri, distortion, rvec, tvec):
        try:
            self.frame = cv2.aruco.drawAxis(
                self.frame, intri, distortion, rvec, tvec, 10
            )
        except Exception as e:
            print(e)
        return self.frame

    def keyboard(self, key):
        # print("key:", key)
        fb_speed = 50
        lf_speed = 50
        ud_speed = 80
        degree = 10
        if key == ord("1"):
            self.takeoff()
            self.move_up(50)
        if key == ord("2"):
            self.land()
        if key == ord("3"):
            self.stop()
            print("stop!!!!")
        if key == ord("w"):
            self.move_forward(fb_speed)
            print("forward!!!!")
        if key == ord("s"):
            self.move_backward(fb_speed)
            print("backward!!!!")
        if key == ord("a"):
            self.move_left(lf_speed)
            print("left!!!!")
        if key == ord("d"):
            self.move_right(lf_speed)
            print("right!!!!")
        if key == ord("z"):
            self.move_up(ud_speed)
            print("up!!!!")
        if key == ord("x"):
            self.move_down(ud_speed)
            print("down!!!!")
        if key == ord("c"):
            self.move(0, 0, 0, degree)
            print("rotate!!!!")
        if key == ord("v"):
            self.move(0, 0, 0, -degree)
            print("counter rotate!!!!")
        if key == ord("5"):
            height = self.get_height()
            print(height)
        if key == ord("6"):
            battery = self.get_battery()
            print(battery)
        time.sleep(0.1)
        self.stop()


def frame_update(drone):
    while True:
        frame = drone.get_cur_frame()
        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)
        drone.keyboard(key)
        time.sleep(0.01)


def main():
    # Tello
    drone = Drone()
    drone.connect()
    drone.adjust()

    # Thread
    frame_thread = threading.Thread(target=frame_update, args=(drone,))
    frame_thread.start()
    frame_thread.setDaemon(True)

    while True:
        # img
        frame = drone.get_frame()

        # Load the predefined dictionary
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

        # Initialize the detector parameters using default values
        parameters = cv2.aruco.DetectorParameters_create()

        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
            frame, dictionary, parameters=parameters
        )

        frame = drone.draw_detected_markers()
        if len(markerCorners) > 0:
            global rvec, tvec
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
                markerCorners, 15, intri, distortion
            )

            # MAX_SPEED_THRESHOLD = 25
            # z_update = tvec[0, 0, 2] - 100
            # # print("org_z: " + str(z_update))
            # z_update = z_pid.update(z_update, sleep=0)
            # # print("pid_z: " + str(z_update))
            # if z_update > MAX_SPEED_THRESHOLD:
            #     z_update = MAX_SPEED_THRESHOLD
            # elif z_update < -MAX_SPEED_THRESHOLD:
            #     z_update = -MAX_SPEED_THRESHOLD

            # # drone.send_rc_control(0, int(z_update), 0, 0)

            # x_update = 0
            # if tvec[0, 0, 0] > 15:
            #     x_update = 10
            # elif tvec[0, 0, 0] < -15:
            #     x_update = -10

            # #drone.send_rc_control(x_update, 0, 0, 0)

            # R, _ = cv2.Rodrigues(rvec)
            # Z = np.array([0,0,1])
            # Z_prime = np.dot(R, Z)
            # V = np.array([Z_prime[0], 0, Z_prime[2]])
            # angle_rad = math.atan2(np.linalg.norm(np.cross(Z, V)), np.dot(Z, V))
            # angle_deg = math.degrees(angle_rad)

            # deg = 0
            # deg_mv = 0

            # if R[2, 0] > 0.1:
            #     deg = -10
            #     deg_mv = 5
            # elif R[2, 0] < -0.1:
            #     deg = 10
            #     deg_mv = 5

            # drone.send_rc_control(x_update, int(z_update), 0, deg)

            frame = drone.draw_axis(intri, distortion, rvec, tvec)
            cv2.putText(
                frame,
                f"x: {round(tvec[0,0,0], 2)}, y: {round(tvec[0,0,1], 2)}, z: {round(tvec[0,0,2], 2)}, deg: {R[2, 0]}",
                (0, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            print(
                f"x: {round(tvec[0,0,0], 2)}, y: {round(tvec[0,0,1], 2)}, z: {round(tvec[0,0,2], 2)}, deg: {round(R[2, 0], 2)}"
            )

        if len(markerCorners) > 0:
            print(f"ID: {markerIds}")
            print(f"Progress: {drone.progress}")
            if [1] in markerIds and drone.progress == 0:
                z = tvec[0, 0, 2]
                if z > 100:
                    x_update = 0
                    if tvec[0, 0, 0] > 15:
                        x_update = 20
                    elif tvec[0, 0, 0] < -15:
                        x_update = -20
                    if tvec[0, 0, 1] > -5:
                        y_update = -10
                    elif tvec[0, 0, 1] < -15:
                        y_update = 10
                    # R, _ = cv2.Rodrigues(rvec)
                    # Z = np.array([0,0,1])
                    # Z_prime = np.dot(R, Z)
                    # V = np.array([Z_prime[0], 0, Z_prime[2]])
                    # angle_rad = math.atan2(np.linalg.norm(np.cross(Z, V)), np.dot(Z, V))
                    # angle_deg = math.degrees(angle_rad)

                    # deg = 0
                    # deg_mv = 0

                    # if R[2, 0] > 0.1:
                    #     deg = -10
                    #     deg_mv = 5
                    # elif R[2, 0] < -0.1:
                    #     deg = 10
                    #     deg_mv = 5
                    # drone.send_rc_control(x_update, 10, y_update, 0)
                    # time.sleep(0.5)
                    # stop(drone)
                    drone.move(x_update, 10, y_update, 0)
                else:
                    drone.stop()
                    drone.progress = 1
            if drone.progress == 1:
                INTERVAL = 1
                if [1] in markerIds <= 100:
                    # up 120cm
                    SPEED = 50
                    for _ in range(int(150 / SPEED)):
                        print("Up")
                        drone.move_up(SPEED)
                    drone.stop()
                    # forward 90cm
                    SPEED = 45
                    for _ in range(int(90 / SPEED)):
                        print("Forward")
                        drone.move_forward(SPEED)
                    drone.stop()
                    # down 120cm
                    SPEED = 60
                    for _ in range(int(180 / SPEED)):
                        print("Down")
                        drone.move_down(SPEED)
                    drone.stop()
                    drone.progress = 2
            if drone.progress == 2:
                INTERVAL = 1
                SPEED = 60
                if [2] in markerIds and [1] not in markerIds:
                    # down 60cm
                    for _ in range(int(60 / SPEED)):
                        print("Down")
                        drone.move_down(SPEED)
                    drone.stop()
                    # forward 180cm
                    for _ in range(int(180 / SPEED)):
                        print("Forward")
                        drone.move_forward(SPEED)
                    drone.stop()
                    drone.land()

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
