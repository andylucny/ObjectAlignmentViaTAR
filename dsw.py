import numpy as np

def DSW(s, t, shift=0, window=5):
    n, m = len(s), len(t)

    w = np.max([window, abs(n-m)])
    dsw_matrix = np.zeros((n+1, 2*w+3), np.double)
    path_matrix = np.zeros((n+1, 2*w+3), np.int)
    
    dsw_matrix[:,:] = np.inf
    path_matrix[:,:] = -1
    dsw_matrix[0,w+1] = 0
    
    for i0 in range(1, n+1):
        i = i0-1-shift
        if i < 0:
            i += n
        for j0 in range(1,2*w+2):
            j = i0+j0-w-2
            if j >= 0 and j < m:
                cost = np.sum(np.abs(s[i] - t[j])) / len(s[i])
                choices = [dsw_matrix[i0-1, j0], dsw_matrix[i0-1, j0+1], dsw_matrix[i0, j0-1]]
                dsw_matrix[i0, j0] = cost + np.min(choices)
                path_matrix[i0, j0] = np.argmin(choices)
    
    cost = dsw_matrix[n, w+1]
    path = []
    i0, j0 = n, w+1
    while path_matrix[i0, j0] != -1:
        i = i0-1-shift
        if i < 0:
            i += n
        j = i0+j0-w-2
        path.append((i,j))
        i0, j0 = [(i0-1, j0), (i0-1, j0+1), (i0, j0-1)][path_matrix[i0, j0]]
        
    return cost, path[::-1]

def cyclicDSW(s, t, window=5, percentage=1, nms_radius=-1):
    n = len(s)
    window = window if window > 0 else n

    costs = [0 for i in range(n)]
    paths = [[] for i in range(n)]
    for shift in range(n):
        costs[shift], paths[shift] = DSW(s, t, shift, window)
    
    limit = np.min(costs)
    limit += (np.max(costs)-limit)*percentage/100 + 1e-5
    #print('limit',limit)
    best_costs = []
    best_paths = []
    best_shifts = []
    indices = np.argsort(costs)
    nms_radius = window if nms_radius < 0 else nms_radius
    #print('nms',nms_radius)
    supressed = np.zeros(indices.shape,np.bool)
    for index in indices:
        if costs[index] > limit: 
            break
        if not supressed[index]:
            best_costs.append(costs[index])
            best_paths.append(paths[index])
            best_shifts.append(index)
            supressed[np.arange(index-nms_radius,index+nms_radius+1) % indices.shape] = True

    return best_costs, best_paths, best_shifts
    
# Test
from d11612102066 import d1 
from d21612102066 import d2
a=d1()[:,:10]
b=d2()[:,:10]

costs, pairings, shifts = cyclicDSW(a,b,window=5,nms_radius=-1,percentage=20)
