import cv2
import numpy as np

# Open a connection to the webcam (usually 0 or 1 for built-in webcams)
cap = cv2.VideoCapture(0)

# Load the background image
background = cv2.imread('bg.jpg')

# Define the coordinates of the four corners of your webcam frame and the corresponding coordinates on the background
# webcam_corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
# screen_corners = np.array([[0, 0], [background.shape[1], 0], [background.shape[1], background.shape[0], [0, background.shape[0]]], dtype=np.float32)
webcam_corners = np.array([[276, 189], [618, 86], [263, 412], [627, 386]], dtype=np.float32)
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    print("read")
    h, w, _ = frame.shape
    screen_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)

    # Calculate the perspective transform matrix using getPerspectiveTransform
    perspective_matrix = cv2.getPerspectiveTransform(webcam_corners, screen_corners)

    # Create a canvas of the same size as the background
    # background = np.zeros_like(background, dtype=np.uint8)

    # Iterate through each pixel in the output image and perform bilinear interpolation
    for y in range(background.shape[0]):
        for x in range(background.shape[1]):
            point = np.array([[x, y]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), perspective_matrix)
            x_transformed, y_transformed = transformed_point[0][0]

            # Check if the transformed coordinates are within the bounds of the webcam frame
            if 0 <= x_transformed < frame.shape[1] - 1 and 0 <= y_transformed < frame.shape[0] - 1:
                x0, y0 = int(x_transformed), int(y_transformed)
                x1, y1 = x0 + 1, y0 + 1

                # Perform bilinear interpolation
                dx = x_transformed - x0
                dy = y_transformed - y0
                background[y, x] = (1 - dx) * (1 - dy) * frame[y0, x0] + dx * (1 - dy) * frame[y0, x1] + (1 - dx) * dy * frame[y1, x0] + dx * dy * frame[y1, x1]

    # Display the output
    cv2.imshow('Mapped Webcam Frame', background)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
