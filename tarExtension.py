import numpy as np
import cv2

def triangleValidArea(binary, A, B, C):
    rect = cv2.boundingRect(np.array([A,B,C],np.int))
    tl = np.array([rect[0],rect[1]],np.int)
    rendered = np.zeros((rect[3],rect[2]),np.uint8)
    cv2.drawContours(rendered, np.array([[[A - tl], [B - tl], [C - tl]]]), 0, 255, cv2.FILLED)
    roi = binary[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    rendered = np.bitwise_and(roi,rendered)
    return float(cv2.countNonZero(rendered))

def TARextension(resampled, binary, triangleSideLength=10):
    N = len(resampled)
    center = np.average(resampled,axis=0).astype(np.int)
    w = triangleSideLength
    d = np.zeros(N,np.float)
    for i in range(N):
        p0 = resampled[(i-w-1)%N]
        p2 = resampled[(i+w+1)%N]
        d[i] = triangleValidArea(binary, p0, p2, center)
    maximum = np.max(d)
    if maximum > 1e-5:
        d /= maximum
    return d;

