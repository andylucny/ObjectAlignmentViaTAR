import numpy as np
import cv2
import matplotlib.pyplot as plt
from rigidTransform import findRigidTransform

def findAlignment(a,b):
    indices = np.arange(len(a))
    changes = 1
    while changes > 0:
        R, t = findRigidTransform(a,b[indices])
        c = a @ R.T + t
        changes = 0
        for i in range(len(a)):
            previous = indices[i]
            indices[i] = np.argmin(np.linalg.norm(b-c[i],axis=1))
            if indices[i] != previous:
                changes += 1
    return R, t

