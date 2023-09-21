import cv2

img = cv2.imread("nctu_flag.jpg")
row, col, _ = img.shape
# turn not blue pixel to greyscale
for i in range(row):
    for j in range(col):
        p = img[i][j]
        b, g, r = list(map(int, (p[0], p[1], p[2])))
        if not (b * 0.8 > g and b * 0.8 > r):
            # grey scale
            scale = (min(b, g, r) + max(b, g, r)) / 2
            img[i][j] = [scale, scale, scale]

cv2.imwrite("1-1.jpg", img)
