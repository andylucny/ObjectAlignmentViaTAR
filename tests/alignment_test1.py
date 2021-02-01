import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from rigidTransform import findRigidTransform
from alignment import findAlignment

from dst1607107523 import dst as dst_
dst = dst_()
from src1607107523 import src as src_
src = src_()

rect = cv2.boundingRect(np.array(cv2.vconcat([dst[:,:2],src[:,:2]])*20,np.int32))
mini = np.min(cv2.vconcat([dst[:,2],src[:,2]]))
maxi = np.max(cv2.vconcat([dst[:,2],src[:,2]]))

def display(_1,_2):
    disp = np.zeros((rect[3],rect[2],3),np.uint8)
    for i,p in enumerate(np.asarray(_1[:,:2]*20,np.int32)):
        v = 128+int(127*(_1[i,2] - mini)/(maxi-mini+1))
        r = 5 if i==0 else 1
        _ = cv2.circle(disp,(p[0]-rect[0],p[1]-rect[1]),r,(0,0,v),cv2.FILLED)
    for i,p in enumerate(np.asarray(_2[:,:2]*20,np.int32)):
        v = 128+int(127*(_2[i,2] - mini)/(maxi-mini+1))
        r = 5 if i==0 else 1
        _ = cv2.circle(disp,(p[0]-rect[0],p[1]-rect[1]),r,(0,v,0),cv2.FILLED)
    return disp
    
disp = display(dst,src)
cv2.imshow("disp",disp)
cv2.waitKey(0)
cv2.destroyAllWindows()

R, t = findRigidTransform(dst,src)
src1 = dst @ R.T + t

disp = display(src1,src)
cv2.imshow("disp",disp)
cv2.waitKey(0)
cv2.destroyAllWindows()

R, t = findAlignment(dst,src)
src2 = dst @ R.T + t

disp = display(src2,src)
cv2.imshow("disp",disp)
cv2.waitKey(0)
cv2.destroyAllWindows()


