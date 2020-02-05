import numpy as np
import cv2
import matplotlib.pyplot as plt

def RGB2HSV(img):
    img_cp = img.copy()
    img_cp = img_cp / 255.0

    Max = np.max(img_cp, axis=2)
    Min = np.min(img_cp, axis=2)
    arg_min = np.argmin(img_cp, axis=2)
    # BGR
    B = img_cp[:,:,0].copy()
    G = img_cp[:,:,1].copy()
    R = img_cp[:,:,2].copy()

    # H
    img_cp[:, :, 0] = np.where(Max==Min, np.zeros(Max.shape), img_cp[:, :, 0])
    img_cp[:, :, 0] = np.where(arg_min==0, 60.0*(G-R)/(Max-Min) + 60, img_cp[:, :, 0])
    img_cp[:, :, 0] = np.where(arg_min==2, 60.0*(B-G)/(Max-Min) + 180, img_cp[:, :, 0])
    img_cp[:, :, 0] = np.where(arg_min==1, 60.0*(R-B)/(Max-Min) + 300, img_cp[:, :, 0])
    img_cp[:, :, 0] = img_cp[:, :, 0] % 360

    # S
    img_cp[:, :, 1] = Max-Min

    # V
    img_cp[:, :, 2] = Max

    return img_cp

def HSV2RGB(img):

    H = img[:, :, 0].copy()
    S = img[:, :, 1].copy()
    V = img[:, :, 2].copy()

    Hp = H / 60.0
    X = S * (1 - np.abs(Hp % 2 - 1))
    cond = np.stack([Hp, Hp, Hp], axis=2)
    zero = np.zeros(np.shape(S))

    base = np.stack([V-S, V-S, V-S], axis=2)
    imgRGB = np.where(cond==0, base+np.stack([zero,zero,zero], axis=2), base)
    imgRGB = np.where(cond>0, base+np.stack([S,X,zero], axis=2), imgRGB)
    imgRGB = np.where(cond>=1, base+np.stack([X,S,zero], axis=2), imgRGB)
    imgRGB = np.where(cond>=2, base+np.stack([zero,S,X], axis=2), imgRGB)
    imgRGB = np.where(cond>=3, base+np.stack([zero,X,S], axis=2), imgRGB)
    imgRGB = np.where(cond>=4, base+np.stack([X,zero,S], axis=2), imgRGB)
    imgRGB = np.where(cond>=5, base+np.stack([S,zero,X], axis=2), imgRGB)

    imgBGR = imgRGB[:,:,(2,1,0)]

    return imgBGR*255.0

def invert_Hue(img):
    imgHSV = RGB2HSV(img)
    imgHSV[:,:,0] = (imgHSV[:,:,0] + 180) % 360
    return HSV2RGB(imgHSV)

img = cv2.imread("imori.jpg")
# cv2.imshow("imori", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = RGB2HSV(img)
# img = HSV2RGB(img)

# img = img.astype(np.uint8)
# cv2.imshow("imori", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img = invert_Hue(img)
cv2.imshow("imori", img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

