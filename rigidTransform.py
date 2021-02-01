import numpy as np
import cv2

#Kabsch algorithm
def findRigidTransform(points1, points2):
    t1 = -np.average(points1,axis=0)
    t2 = -np.average(points2,axis=0)
    
    T1 = np.eye(4)
    T2 = np.eye(4)
    T1[:3,3] = t1
    T2[:3,3] = -t2

    C = np.zeros((3,3))
    p1 = points1 + t1
    p2 = points2 + t2
    C = np.sum([p2[i].reshape(3,1)@p1[i].reshape(3,1).T for i in range(len(p2))],axis=0)
    s,u,v = cv2.SVDecomp(C)
    I = np.eye(3)
    if np.linalg.det(v.T @ u.T) < 0:
        I[2,2] = -I[2,2]
    
    R = u @ I @ v
    
    M = np.eye(4)
    M[:3,:3] = R

    result = M / M[3, 3]
    result = T2 @ result @ T1
    result = result[:3]
    return result[:3,:3], result[:,3]

