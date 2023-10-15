import numpy as np
import cv2 as cv
img = cv.imread('otsu.jpg', cv.IMREAD_GRAYSCALE)
row, col = img.shape

max_between_var = {'intensity':0, 'variance':0.0}
hist = np.zeros(256)
def var_between(Nb, Mb, No, Mo):
    return Nb*No*((Mb - Mo)**2)

for i in range(row):
    for j in range(col):
        hist[img[i][j]] += 1;
#initialize nb, no, mb, mo
nb = hist[0]
no = (row*col) - hist[0]
mb = 0
mo = 0
for i in range(1, 256):
    mo += hist[i]*i
    mo = round(float(mo)/((row*col) - nb))
max_between_var['intensity'] = 0
max_between_var['variance'] = var_between(nb, mb, no, mo)
#print(f'mb: {mb}, mo: {mo}, nb: {nb}, no: {no} i: {i} var: {max_between_var["variance"]}')

temp = 0
for i in range(1, 256):
    mb = (mb*nb + hist[i]*i)/(nb + hist[i])
    mo = (mo*no - hist[i]*i)/(no - hist[i])
    nb += hist[i]
    no -= hist[i]
    if mo <= 0 :
        continue
    temp = var_between(nb, mb, no, mo)
   # print(f'mb: {mb}, mo: {mo}, nb: {nb}, no: {no} i: {i} var: {temp}')
    if (temp > max_between_var['variance']):
        max_between_var['intensity'] = i
        max_between_var['variance'] = temp
for i in range(row):
    for j in range(col):
        if (img[i][j] > max_between_var['intensity']):
            img[i][j] = 255
        else:
            img[i][j] = 0
cv.imshow('3.jpg', img)
cv.waitKey(0)
#print(max_between_var)

