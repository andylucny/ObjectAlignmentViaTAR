import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from rigidTransform import findRigidTransform

points1 = np.array([[1,0,1],[0,0,1],[0,1,1]]) 
points2 = np.array([[0,1,1],[0,0,1],[-1,0,1]])

warp = findRigidTransform(points1, points2)

transformed1 = np.insert(points1, 3, values=1, axis=1) @ warp.T

plt.fill(points1[:,0],points1[:,1],color='g',fill=True)
plt.fill(points2[:,0],points2[:,1],color='r',fill=True)
plt.fill(transformed1[:,0],transformed1[:,1],edgecolor='b',fill=False)
plt.show()

plt.fill(points1[:,2],points1[:,1],color='g',fill=True)
plt.fill(points2[:,2],points2[:,1],color='r',fill=True)
plt.fill(transformed1[:,2],transformed1[:,1],edgecolor='b',fill=False)
plt.show()

