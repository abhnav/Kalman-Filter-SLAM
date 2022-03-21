import numpy as np

# CONSTANTS DEFINED
# Projection matrix
P = np.zeros((3,4))
P[:3,:3] = np.eye(3)

# Control Noise covariance
W = np.zeros((6,6)) 
W[0,0] = W[1,1] = W[2,2] = 0.3
W[3,3] = W[4,4] = W[5,5] = 0.05

# Stereo camera Noise covariance
V = np.eye(4)

