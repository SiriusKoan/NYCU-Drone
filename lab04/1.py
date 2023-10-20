import numpy as np
import cv2

# 棋盤格規格，假設 6x9 的內部角點
pattern_size = (9, 6)

# 創建 object points，假設棋盤格在平面上
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# 儲存 object points 和 image points 的列表
obj_points = []  # 存放 object points
img_points = []  # 存放 image points

# 開啟攝像機
print("Turning on camera")
cap = cv2.VideoCapture(1)  # 0代表默認攝像機，根據需要更改
print("Starting reading.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

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
    if len(img_points) > 100:
        print("Collection done")
        break

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 進行相機校準
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# 將相機內部參數儲存到XML檔
cv2_file = cv2.FileStorage("camera_params.xml", cv2.FileStorage_WRITE)
cv2_file.write("camera_matrix", mtx)
cv2_file.write("dist_coeff", dist)
cv2_file.release()
