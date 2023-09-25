import sys
import cv2
import numpy as np

filename = sys.argv[1]
rate = int(sys.argv[2])

# filename: test.jpg
# rate: 3 

img = cv2.imread(filename)
row, col, _ = img.shape

# new image np array
new_img = np.zeros((row * rate, col * rate, 3), np.uint8)

for i in range(row):
    for j in range(col):
        p = img[i][j]
        b, g, r = list(map(int, (p[0], p[1], p[2])))
        for k in range(rate):
            for l in range(rate):
                new_img[i * rate + k][j * rate + l] = [b, g, r]

cv2.imwrite("2.jpg", new_img)
