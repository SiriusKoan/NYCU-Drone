import numpy as np
import cv2 as cv

img = cv.imread('otsu.jpg',cv.IMREAD_GRAYSCALE)
row, col = img.shape
total_pixels = row*col
# histogram
hist = np.zeros(256)

for i in range(row):
    for j in range(col):
        hist[img[i][j]] += 1
# calculte best threshold
best_threshold = 0
max_variance = 0

for threshold in range(256):
    # < threshold
    pixels_below = np.sum(hist[:threshold])
    # >= threshold
    pixels_above = total_pixels - pixels_below

    if (pixels_below == 0 or pixels_above == 0):
        continue

    mean_below = np.sum(np.arange(threshold)*hist[:threshold])/float(pixels_below)
    mean_above = np.sum(np.arange(threshold, 256)*hist[threshold:])/float(pixels_above)

    between_variance = pixels_above*pixels_below*(mean_above - mean_below)**2
    #print(f'intensity: {threshold}, pixels_below: {pixels_below}, pixels_above: {pixels_above}, mean_below: {mean_below}, mean_above: {mean_above}, between_variance: {between_variance}')
    if (between_variance > max_variance):
        max_variance = between_variance
        best_threshold = threshold

for i in range(row):
    for j in range(col):
        if (img[i][j] < best_threshold):
            img[i][j] = 0
        else:
            img[i][j] = 255

#_, new_img = cv.threshold(img, best_threshold, 255, cv.THRESH_BINARY)
cv.imwrite('3.jpg', img)
cv.imshow('3.jpg', img)
cv.waitKey(0)
