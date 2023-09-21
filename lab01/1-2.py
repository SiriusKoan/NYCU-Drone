import numpy as np
import cv2

img = cv2.imread("nctu_flag.jpg")
row, col, _ = img.shape
CONTRAST = 100
BRIGHTNESS = 40

# turn image pixel into int32
img = img.astype(np.int32)

for i in range(row):
    for j in range(col):
        p = img[i][j]
        b, g, r = list(map(int, (p[0], p[1], p[2])))
        if (b + g) * 0.3 > r:
            # high contrast
            img[i][j] = (img[i][j] - 127) * (CONTRAST / 127 + 1) + 127 + BRIGHTNESS

cv2.imwrite("1-2.jpg", img)
