import numpy as np
import cv2

img = cv2.imread("imori.jpg")

img = img * np.array([0.0722, 0.7152, 0.2126])
img = np.sum(img, axis=2)
print(img)

cv2.imshow("imori", img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
