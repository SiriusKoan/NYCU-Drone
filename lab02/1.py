import numpy as np 
import cv2 as cv

img = cv.imread('Lenna.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (5, 5), 0)

sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0 ,-1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
grad_x = cv.filter2D(img, -1, sobel_x)
grad_y = cv.filter2D(img, -1, sobel_y)
grad = cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

cv.imwrite('1.jpg', grad)
cv.imshow('1.jpg', grad)
cv.waitKey(0)

