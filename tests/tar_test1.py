import numpy as np
import sys

sys.path.append("..")
from tar import resampleContour, TARdescriptor, TARdistances, TARpoints

c = [(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)]
c2 = c[2:]+c[:2]

r,indices = resampleContour(c,4)
r2,indices2 = resampleContour(c2,4)

d,r,indices = TARdescriptor(c,2,4)
d2,r2,indices2 = TARdescriptor(c2,2,4)
costs, paths, shifts = TARdistances(d,d2,window=2,nms_radius=0)
points = TARpoints(c,indices,paths[3],0)
points2 = TARpoints(c2,indices2,paths[3],1)

