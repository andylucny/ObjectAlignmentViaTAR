import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from retinex import isodataThreshold

image = np.asarray(np.random.rand(100,100)*255,np.uint8)
threshold = isodataThreshold(image)
print(threshold)

image = np.zeros((100,100),np.uint8)
threshold = isodataThreshold(image)
print(threshold)

image = np.ones((100,100),np.uint8)*255
threshold = isodataThreshold(image)
print(threshold)

image = np.ones((100,100),np.uint8)*255
image[0,0]=0
threshold = isodataThreshold(image)
print(threshold)

from retinex import retinexBinarization
import cv2

a = np.zeros((500,500),np.float32)
a[:,:] = np.nan
a[100:400,100:400] = 0.0
a[150:370,150:350] = 1.0
a[200:290,210:300] = 2.0
a[240:260,240:250] = 3.0
a[120:130,115:130] = np.nan
b, _, _ = retinexBinarization(a)
cv2.imshow('retinex',b)
cv2.waitKey(0)

a = np.zeros((500,500),np.float32)
a[:,:] = np.nan
a[100:400,100:400] = 1.0
a[200:300,100:400] = 0.0
for i in range(50):
    a[200-i,100:400] = i/50.0
    a[300+i,100:400] = i/50.0
b, _, _ = retinexBinarization(a)
cv2.imshow('retinex',b)
cv2.waitKey(0)

a = np.zeros((500,500),np.float32)
a[:,:] = np.nan
a[100:400,100:400] = 1.0
b, _, _ = retinexBinarization(a)
cv2.imshow('retinex',b)
cv2.waitKey(0)

cv2.destroyAllWindows()
