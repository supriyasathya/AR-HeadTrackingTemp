
#script to compute error or change in pose between two transformation matrices. In the accuracy measurement experiments, A1 is the pose of a cube before and A2 is the pose after any manual adjustment of the virtual rendering to align with the real-world counterpart.
#Replace A1 and A2 with transoformation matrices we want to find the error between.


import numpy as np 
from math import sqrt 
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R


A1 = np.array([ 0.0004996305,0.0001996768,-4.580941E-05,0.123648,4.298564E-06,0.000110504,0.000528555,-0.2485259,0.0002048191,-0.0004894058,0.0001006534,0.7053052,0,0,0,1])
A2 = np.array([0.0005108861,0.0001710876,-3.639199E-05,0.1272283,5.331357E-06,9.706646E-05,0.0005311777,-0.2483861,0.000174834,-0.0005028988,9.014404E-05,0.7108862,0,0,0,1])
A1 = A1.reshape(4,4)
A2 = A2.reshape(4,4)


A1rot = A1[:3,:3]
A2rot = A2[:3,:3]

A1tra = A1[:3,3]
A2tra = A2[:3,3]


A1R = R.from_matrix(A1rot)
A2R = R.from_matrix(A2rot)

A1euler = A1R.as_euler('xyz', degrees=True)
A2euler = A2R.as_euler('xyz',degrees= True)

rmsrot = sqrt(mean_squared_error(A1euler,A2euler))
rmstra = sqrt(mean_squared_error(A1tra,A2tra))
print(A1euler, A2euler)
print(A1tra, A2tra)

print("rmsrot", rmsrot)
print("rmstra", rmstra)


