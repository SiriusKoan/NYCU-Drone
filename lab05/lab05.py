import cv2
import numpy as np
import time
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
    

    while True:
        frame = frame_read.frame
        # success, frame = vid.read()

                
        # Load the predefined dictionary
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

        # Initialize the detector parameters using default values
        parameters = cv2.aruco.DetectorParameters_create()

        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)


        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # print(intri, distortion)
        if len(markerCorners) > 0:
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intri, distortion)
            print(rvec, tvec)
            frame = cv2.aruco.drawAxis(frame, intri, distortion, rvec, tvec, 10)
            cv2.putText(frame,
                        "x = "+str(round(tvec[0,0,0], 4))+", y = "+str(round(tvec[0,0,1], 4))+", z = "+str(round(tvec[0,0,2], 4)),
                        (0,64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2,
                        cv2.LINE_AA
            )
        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)
        time.sleep(0.01)
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    #check_lens()
    main()


