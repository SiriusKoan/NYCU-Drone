import cv2
import numpy as np

# Known parameters
known_person_width = 0.5  # Width of the person in meters (example value)
known_person_height = 1.7  # Height of the person in meters (example value)
known_face_width = 0.15   # Width of the face in meters (example value)
known_face_height = 0.2   # Height of the face in meters (example value)
focal_length = 25000       # Focal length of the camera (example value)

# 開啟攝像機
print("Turning on camera")
cap = cv2.VideoCapture(0)  # 0代表默認攝像機，根據需要更改
print("Starting reading.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(frame, winStride=(6, 6), padding=(8, 8), scale=1.05, useMeanshiftGrouping=False)

    # draw the bounding boxes for people, yellow
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

        # Calculate depth for people
        depth = (known_person_width * known_person_height * focal_length) / (w * h)
        cv2.putText(frame, f'Depth: {depth:.2f} meters', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # draw the bounding boxes for faces, green
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Calculate depth for faces
        depth = (known_face_width * known_face_height * focal_length) / (w * h)
        cv2.putText(frame, f'Depth: {depth:.2f} meters', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Calibration', frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
