import sys
import cv2
import numpy as np

# read the image and select ROI 
img = cv2.imread('test.jpg')
rect = cv2.selectROI(img)

cv2.destroyAllWindows()

# num of iterations
iter_num = int(sys.argv[1])

# initialize background and foreground models
b_Model = np.zeros((1,65),np.float64)
f_Model = np.zeros((1,65),np.float64)

# grabCut algorithm
mask_new, b_model, f_model=cv2.grabCut(img, None, rect, b_Model, f_Model, iter_num, cv2.GC_INIT_WITH_RECT) 

# use mask to extract the foreground
mask = np.where((mask_new==0)|(mask_new==2),0,1).astype('uint8')
new_img = img*mask[:,:,np.newaxis]

# show the new image
cv2.imwrite("1.jpg", new_img)
cv2.imshow('New image', new_img)
cv2.waitKey(0)

cv2.destroyAllWindows()