import cv2

# capture the video
cap = cv2.VideoCapture("train.mp4")

# create a backgroundSubtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# check if the video was successfully captured
if not cap.isOpened():
    print("cannot capture the video")
    exit()

while True:
    # ret: boolean value that 'True' if a frame was read, 'False' if the end of the video is reached or an error occurred
    # frame: video frame that was read
    ret, frame = cap.read()

    # check if video is reached or an error occurred
    if not ret:
        break

    # apply the backgroundSubtractor to the current frame to detect moving objects
    fgmask = backSub.apply(frame)

    # get the shadow value
    shadowval = backSub.getShadowValue()

    # threshold the foreground mask
    ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)

    # find connected components in nmask
    # stats: x, y, WIDTH, HEIGHT, AREA
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        nmask, connectivity=8
    )

    # define the threshold of area
    area_T = 600

    # iterate over the connected components
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # if the area exceeds threshold, get the position of the connected component and draw a green rectangle
        if area > area_T:
            x, y, w, h, _ = stats[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the video frame per 33ms
    cv2.imshow("frame", frame)
    key = cv2.waitKey(33)

    # press 'esc' to quit
    if key == 27:
        break
    if key == 32:
        key = cv2.waitKey()
        continue

cap.release()
cv2.destroyAllWindows()
