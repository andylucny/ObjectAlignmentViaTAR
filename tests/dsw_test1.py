import numpy as np
import sys

sys.path.append("..")
from dsw import DSW, cyclicDSW

from d11612102066 import d1 
from d21612102066 import d2
a=d1()[:,:10]
b=d2()[:,:10]

costs, pairings, shifts = cyclicDSW(a,b,window=5,nms_radius=-1,percentage=20)
