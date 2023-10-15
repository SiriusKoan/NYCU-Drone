import numpy as np
import cv2 as cv

img = cv.imread('histogram.jpg')
row, col, _ = img.shape
new_img = np.zeros((row, col, 3), np.uint8)
hist_b = np.zeros(256)
hist_g = np.zeros(256)
hist_r = np.zeros(256)

for i in range(row):
    for j in range(col):
        hist_b[img[i][j][0]]+= 1;
        hist_g[img[i][j][1]]+= 1;
        hist_r[img[i][j][2]]+= 1;
        
sum_b = 0
sum_g = 0
sum_r = 0
for i in range(256):
    sum_b += float(hist_b[i])/(row*col)
    hist_b[i] = round(sum_b*255)

    sum_g += float(hist_g[i])/(row*col)
    hist_g[i] = round(sum_g*255)

    sum_r += float(hist_r[i])/(row*col)
    hist_r[i] = round(sum_r*255)
for i in range(row):
    for j in range(col):
        new_img[i][j][0] = hist_b[img[i][j][0]] 
        new_img[i][j][1] = hist_b[img[i][j][1]] 
        new_img[i][j][2] = hist_b[img[i][j][2]] 
cv.imwrite('2-a.jpg', new_img)
cv.imshow('2-a', new_img)
cv.waitKey(0)


        

    
