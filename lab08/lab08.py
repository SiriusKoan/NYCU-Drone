import cv2
import numpy as np

# Known parameters
known_person_width = 0.5  # Width of the person in meters (example value)
known_person_height = 1.7  # Height of the person in meters (example value)
known_face_width = 0.15   # Width of the face in meters (example value)
known_face_height = 0.2   # Height of the face in meters (example value)
focal_length = 25000       # Focal length of the camera (example value)

# Chessboard pattern size for calibration
pattern_size = (6, 9)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in the image plane
calibrated = False

# Camera calibration parameters
camera_matrix = None
dist_coeffs = None

# Generate object points for the chessboard
objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Open the camera
print("Turning on camera")
cap = cv2.VideoCapture(0)  # 0 represents the default camera, change as needed
print("Starting reading.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if not calibrated:
        # Find chessboard corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
            objpoints.append(objp)
            imgpoints.append(corners)

            # Calibrate camera using the collected points
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            calibrated = True

    else:
        # Undistort the frame using cameraMatrix and distCoeffs
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # initialize the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(
            undistorted_frame, winStride=(6, 6), padding=(8, 8), scale=1.05, useMeanshiftGrouping=False
        )
        #focal length
        #focal_length = camera_matrix[0, 0]

        # draw the bounding boxes for people, yellow
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

            # Calculate depth for people
            depth = (known_person_height * focal_length) / (h*100)
            cv2.putText(frame, f'Depth: {depth:.2f} meters', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        rects = face_cascade.detectMultiScale(undistorted_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # draw the bounding boxes for faces, green
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Calculate depth for faces
            depth = (known_face_height * focal_length) / (h*100)
            cv2.putText(frame, f'Depth: {depth:.2f} meters', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Calibration', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
