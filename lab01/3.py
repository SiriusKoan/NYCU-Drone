import sys
import cv2
import numpy as np

filename = sys.argv[1]
rate = int(sys.argv[2])
img = cv2.imread(filename)
row, col, _ = img.shape

new_img = np.zeros((row * rate, col * rate, 3), np.uint8)

# bilinear interpolation
for i in range(row):
    print(f"{i}/{row}")
    for j in range(col):
        p = img[i][j]
        b, g, r = list(map(int, (p[0], p[1], p[2])))
        for k in range(rate):
            for l in range(rate):
                x = i * rate + k
                y = j * rate + l
                p1 = img[x // rate][y // rate]
                b1, g1, r1 = list(map(int, (p1[0], p1[1], p1[2])))
                if y // rate + 1 >= col:
                    p2 = img[x // rate][y // rate]
                else:
                    p2 = img[x // rate][y // rate + 1]
                b2, g2, r2 = list(map(int, (p2[0], p2[1], p2[2])))
                if x // rate + 1 >= row:
                    p3 = img[x // rate][y // rate]
                else:
                    p3 = img[x // rate + 1][y // rate]
                b3, g3, r3 = list(map(int, (p3[0], p3[1], p3[2])))
                if x // rate + 1 >= row or y // rate + 1 >= col:
                    p4 = img[x // rate][y // rate]
                else:
                    p4 = img[x // rate + 1][y // rate + 1]
                b4, g4, r4 = list(map(int, (p4[0], p4[1], p4[2])))
                left_x = (x - i * rate) / rate
                right_x = ((i + 1) * rate - x) / rate
                # print(left_x, right_x)
                p1_p3_bgr = tuple(
                    map(
                        sum,
                        zip(
                            tuple(c * right_x for c in (b1, g1, r1)),
                            tuple(c * left_x for c in (b3, g3, r3)),
                        ),
                    )
                )
                p2_p4_bgr = tuple(
                    map(
                        sum,
                        zip(
                            tuple(c * right_x for c in (b2, g2, r2)),
                            tuple(c * left_x for c in (b4, g4, r4)),
                        ),
                    )
                )
                upper_y = (y - j * rate) / rate
                lower_y = ((j + 1) * rate - y) / rate
                # print(p1_p3_bgr, p2_p4_bgr)
                tmp = tuple(
                    map(
                        sum,
                        zip(
                            tuple(c * lower_y for c in p1_p3_bgr),
                            tuple(c * upper_y for c in p2_p4_bgr),
                        ),
                    )
                )
                new_img[x][y] = tmp
                assert all([0 <= c <= 255 for c in tmp])

cv2.imwrite("3.jpg", new_img)
