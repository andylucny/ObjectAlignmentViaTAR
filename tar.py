import numpy as np
from dsw import cyclicDSW

def resampleContour(contour, N):
    length = len(contour)
    sqrt2 = np.sqrt(2)
    s = 0
    for p in range(length):
        q = p+1 if p != length-1 else 0
        s += sqrt2 if contour[p][0]!=contour[q][0] and contour[p][1]!=contour[q][1] else 1
    des = s / N
    
    es = s = 0
    resampled = []
    indices = []
    for p in range(length):
        q = p+1 if p != length-1 else 0
        while s >= es:
            resampled.append(contour[p])
            indices.append(p)
            es += des
        s += sqrt2 if contour[p][0]!=contour[q][0] and contour[p][1]!=contour[q][1] else 1
    
    if len(resampled) < N:
        fix = [resampled[0] for i in range(N-len(resampled))]
        resampled = fix + resampled
        fix2 = [0 for i in range(len(fix))]
        indices = fix2 + indices
    
    return resampled, indices

def TARdescriptor(contour, triangleSideLength=10, N=200):
    resampled, indices = resampleContour(contour,N)
    d = np.zeros((N,triangleSideLength),np.float)
    for w in range(triangleSideLength):
        for i in range(N):
            p0 = resampled[(i-w-1)%N]
            p1 = resampled[i]
            p2 = resampled[(i+w+1)%N]
            mat = np.array([[p0[0],p0[1],1],[p1[0],p1[1],1],[p2[0],p2[1],1]])
            d[i,w] = 0.5 * np.linalg.det(mat) 
    maximum = np.max(np.abs(d))
    if maximum > 1e-5:
        d /= maximum
    return d, resampled, indices
    
def TARdistances(descriptorA, descriptorB, window=5, nms_radius=5):
    return cyclicDSW(descriptorA, descriptorB, window=window, nms_radius=nms_radius)
    
def TARpoints(points,indices,path,item):
    return [points[index] for index in [indices[pair[item]] for pair in path]]
