import numpy as np
import cv2
import matplotlib.pyplot as plt


def RGB2Gray(img):
    img_cp = img.copy()
    img_cp = img_cp * np.array([0.0722, 0.7152, 0.2126])
    img_cp = np.sum(img_cp, axis=2)

    return img_cp.astype(np.uint8)

def var_inclass(img, th):
    w0 = img[img<th].size / img.size
    w1 = img[img>=th].size / img.size
    #print(w0, w1)

    return w0*np.var(img[img<th]) + w1*np.var(img[img>=th])

def var_interclass(img, th):
    w0 = img[img<th].size / img.size
    w1 = img[img>=th].size / img.size

    mean0 = np.mean(img[img<th])
    mean1 = np.mean(img[img>=th])
    mean_all = np.mean(img)

    return w0*(mean0 - mean_all)**2 + w1*(mean1-mean_all)**2

def get_opt_th(img):
    ths = np.arange(np.min(img)+1, np.max(img), 1)
    var_bs = [var_interclass(img, th) for th in ths]
    
    return np.argmax(var_bs)

def show_hist(img):
    plt.hist(img.ravel(), bins=255)
    plt.show()

def binarize(img, th):
    img_cp = img.copy()
    img_cp[img_cp<th] = 0
    img_cp[img_cp>=th] = 255
    return img_cp

img = cv2.imread("imori.jpg")
img = RGB2Gray(img)
print(img)
show_hist(img)

var_w = var_inclass(img, 127)
var_b = var_interclass(img, 127)
var = np.var(img)
print("var_w = ", var_w, "var_b = ", var_b, "sum=", var_b+var_w, "overall_var=", var)

th = get_opt_th(img)
print("threshold=", th)
img_bn = binarize(img, th)

cv2.imshow("imori", img_bn)
cv2.waitKey(0)
cv2.destroyAllWindows()
