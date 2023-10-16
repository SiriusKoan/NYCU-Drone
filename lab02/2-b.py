import numpy as np
import cv2 as cv

img = cv.imread('histogram.jpg')
row, col, _ = img.shape
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
new_img = np.zeros((row, col, 3), np.uint8)
hist = np.zeros(256)

for i in range(row):
    for j in range(col):
        hist[img[i][j][2]] += 1
sum = 0
for i in range(256):
    sum += float(hist[i])/(row*col)
    hist[i] = round(sum*255)

new_img = img
for i in range(row):
    for j in range(col):
        new_img[i][j][2] = hist[img[i][j][2]]
new_img = cv.cvtColor(new_img, cv.COLOR_HSV2BGR)
cv.imshow('2-b', new_img)
cv.waitKey(0)
cv.imwrite('2-b.jpg', new_img)


import numpy as np
import cv2 as cv

img = cv.imread('histogram.jpg')
row, col, _ = img.shape
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
new_img = np.zeros((row, col, 3), np.uint8)
hist = np.zeros(256)

for i in range(row):
    for j in range(col):
        hist[img[i][j][2]] += 1
sum = 0
for i in range(256):
    sum += float(hist[i])/(row*col)
    hist[i] = round(sum*255)

new_img = img
for i in range(row):
    for j in range(col):
        new_img[i][j][2] = hist[img[i][j][2]]
new_img = cv.cvtColor(new_img, cv.COLOR_HSV2BGR)
cv.imshow('2-b', new_img)
cv.waitKey(0)
cv.imwrite('2-b.jpg', new_img)


