import numpy as np
import cv2

img = cv2.imread("assets/imori.jpg")

# print(img[:64, :64, 0])
# print(img[:64, :64, 2])
# tmp = img[:64, :64, 0].copy()
# img[:64, :64, 0] = img[:64, :64, 2]
# img[:64, :64, 2] = tmp
# print(img[:64, :64, 0])
# print(img[:64, :64, 2])

img[:64, :64] = img[:64, :64, (2,1,0)]

cv2.imshow("imori", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

