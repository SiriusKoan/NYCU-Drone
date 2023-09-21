import cv2
import numpy as np

img = cv2.imread("nctu_flag.jpg")
row, col, _ = img.shape

# 3x image np array
new_img = np.zeros((row * 3, col * 3, 3), np.uint8)

for i in range(row):
    for j in range(col):
        p = img[i][j]
        b, g, r = list(map(int, (p[0], p[1], p[2])))
        for k in range(3):
            for l in range(3):
                new_img[i * 3 + k][j * 3 + l] = [b, g, r]

cv2.imwrite("2.jpg", new_img)
