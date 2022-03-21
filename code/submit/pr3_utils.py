import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from scipy.linalg import expm, sinm, cosm
from scipy.spatial.transform import Rotation as R

def load_data(file_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
    
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam


def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax

# UTILITY FUNCTIONS DEFINED
def skew(x):
  # Return skew matrix for the given length 3 vector x
  return np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])


def inv_se(tw):
  # SE matrix inversion function
  itw = np.eye(4)
  itw[:3,:3] = tw[:3,:3].T
  itw[:3,3] = -tw[:3,:3].T @ tw[:3,3]
  return itw

def twist_hat(v,w):
  # SE hat for 6 vector = [v,w]
    u_hat = np.zeros((4,4))
    u_hat[:3,:3] = skew(w)
    u_hat[:3, 3] = v
    return u_hat

def proj_diff(p_cam):
  # Derivative of the projection operation
  proj_derivative =  np.eye(4)
  proj_derivative[0,2] = -p_cam[0]/p_cam[2]
  proj_derivative[1,2] = -p_cam[1]/p_cam[2]
  proj_derivative[2,2] = 0
  proj_derivative[3,2] = -p_cam[3]/p_cam[2]
  proj_derivative = (1/p_cam[2]) * proj_derivative
  return proj_derivative

def dot_op(s):
  # dot operation used for pose update
  # s must be homogeneous 4x1 vector
  res = np.zeros((4,6))
  res[:3,:3] = np.eye(3)
  res[:3,3:] = -skew(s[:3])
  return res

def cross_copy(V, Nt):
  # copy 4x4 noise V Nt times along diagonal
  dim = V.shape[0]
  return np.eye(dim*Nt)



