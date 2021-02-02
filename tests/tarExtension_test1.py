import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append("..")
from tar import resampleContour
from tarExtension import TARextension, triangleValidArea

binary = np.zeros((512,512),np.uint8)
cv2.rectangle(binary,(100,100,200,200),255,cv2.FILLED)
cv2.circle(binary,(150,150),30,0,cv2.FILLED)
contours, _ = cv2.findContours(binary,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
points = [(p[0][0],p[0][1]) for p in contours[0]]
resampled, _ = resampleContour(points,200)

#triangleValidArea(binary,(100,100),(100,200),(200,200)) # == 3719.0

extension = TARextension(resampled, binary, triangleSideLength=10)

plt.plot(extension)
plt.show()
