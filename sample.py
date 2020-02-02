import cv2
import numpy as np

img = cv2.imread("assets/imori.jpg")

img2= img.copy().astype(np.float32)
img2[60:100, 60:100, 0] = 260

cv2.imshow("imori", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

img2[np.where(img2 > 255)] = 255

cv2.imshow("imori", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("sample.jpg", img2)
