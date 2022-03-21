import numpy as np
import pdb
from pr3_utils import *
from constants import *
from scipy.linalg import expm, sinm, cosm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm


# MAJOR FUNCTION DEFINITIONS
def joint_se_estimate(T_mu,tau,v,w,joint_sigma,M):
  # Prediction step to return IMU pose and update joint covariance matrix
    v_hat = skew(v)
    w_hat = skew(w)

    u_hat = twist_hat(v,w)

    u_step = tau * u_hat
    u_rot = expm(u_step)
    T_mu = T_mu @ u_rot

    # covariance calculation
    u_chat = np.zeros((6,6))
    u_chat[:3,:3] = w_hat
    u_chat[:3,3:] = v_hat
    u_chat[3:,3:] = w_hat

    u_crot = expm(-tau*u_chat)
    B = joint_sigma[-6:,-6:]
    Bt = u_crot @ B @ u_crot.T + W
    C = joint_sigma[:3*M,-6:]
    Ct = C @ u_crot.T

    joint_sigma[-6:,-6:] = Bt
    joint_sigma[:3*M,-6:] = Ct
    joint_sigma[-6:,:3*M] = Ct.T

    return T_mu

def init_map_landmark(new_lmarks, it, mp, sig, features, T_mu, K, Ks, imu_T_cam, oTi):
  # Initialize the new detected landmarks by using stereo disparity calculation
  for lm in new_lmarks:
    m_z = features[:,lm,it] # observed landmark image position

    disparity = m_z[0] - m_z[2] # u_l - u_r
    if(disparity < 0):
      continue # Bad reading
    landmark_z = (K[0,0] * b) / disparity # depth in optical frame
    if(landmark_z>50): # filter landmarks over 50 meters away
      continue
    landmark_x = landmark_z * (m_z[0] - K[0,2]) / K[0,0]
    landmark_y = landmark_z * (m_z[1] - K[1,2]) / K[1,1]
    mp[lm,0] = landmark_x
    mp[lm,1] = landmark_y
    mp[lm,2] = landmark_z
    mp[lm,3] = 1
    imu_point = imu_T_cam @ mp[lm,:] 
    world_point = T_mu @ imu_point

    mp[lm,:] = world_point
    midx = 3*lm
    sig[midx:midx+3, midx:midx+3] = np.eye(3)

# landmark only update for PartB, since updating each Map point individually
# is faster
# Note: This is not used in the final main function
def update_map_landmark_individually(upd_lmarks, it, mp, mp_sig, features, T_mu, K, Ks, imu_T_cam, oTi):
  for lm in upd_lmarks:
    # print(f"updating!")
    m_z = features[:,lm,it] # observed landmark image position
    m_mu = mp[lm,:] # homogeneous estimated position
    T_mu_inv = inv_se(T_mu)

    midx = 3*lm
    m_sig = mp_sig[midx:midx+3, midx:midx+3] # position variance

    p_cam = oTi @ T_mu_inv.dot(m_mu)
    p_cam = p_cam / p_cam[3] # homogenize

    proj_derivative =  np.eye(4)
    proj_derivative[0,2] = -p_cam[0]/p_cam[2]
    proj_derivative[1,2] = -p_cam[1]/p_cam[2]
    proj_derivative[2,2] = 0
    proj_derivative[3,2] = -p_cam[3]/p_cam[2]
    proj_derivative = (1/p_cam[2]) * proj_derivative

    H = Ks @ proj_derivative @ oTi @ T_mu_inv @ P.T

    p_cam = p_cam / p_cam[2] # projection
    m_z_inn = Ks @ p_cam # expected image position
    innovation = m_z - m_z_inn
    if(abs(innovation).mean() > 20): # if position change by 20 pixels, bad point
      mp[lm,:] = -1 # should we just blacklist the point?
      mp_sig[midx:midx+3, midx:midx+3] = 0

    K_gain = m_sig @ H.T @ np.linalg.inv(H @ m_sig @ H.T + V)
    map_mu_update = mp[lm,:3] + K_gain @ (innovation)
    tt = (np.eye(3) - K_gain @ H)
    map_sig_update =  tt @ m_sig @ tt.T + K_gain @ V @ K_gain.T

    mp[lm,:3] = map_mu_update
    mp_sig[midx:midx+3, midx:midx+3] = map_sig_update

def joint_update_map_pose(upd_lmarks, it, mp, joint_sigma, features, T_mu, K, Ks, imu_T_cam, oTi):
  # Jointly update the landmark positions, IMU pose, Joint Covariance matrix
  threeM = joint_sigma.shape[0] - 6 # 3M  
  lpos = mp[upd_lmarks,:] # Nt x 4 estimated landmark homo positions in world
  observed_image_pos = features[:,upd_lmarks,it] # observed 4xNt img pos
  T_mu_inv = inv_se(T_mu) # 4x4 SE invert matrix
  optical_pos = oTi @ T_mu_inv @ lpos.T # 4 x Nt pose in optical frame
  projected_pos = optical_pos / optical_pos[2,:][None,:]
  predicted_image_pos = Ks @ projected_pos # 4 x Nt img pos in Left, right cam
  jacobian = np.zeros((4*lpos.shape[0], joint_sigma.shape[0])) # 4Nt x 3M+6 jacobian
  for i in np.arange(optical_pos.shape[1]):
    midx = upd_lmarks[i] # index of map point for this jacobian
    pi_derivative = proj_diff(optical_pos[:,i]) # 4x4 projection derivative

    end_term = T_mu_inv @ P.T # 4x3 
    jacobian_i = Ks @ pi_derivative @ oTi @ end_term # 4x3 Jacobian for i
    jacobian[4*i:4*i+4, 3*midx: 3*midx+3] = jacobian_i

    end_term_pose = dot_op(T_mu_inv @ lpos.T[:,i]) # 4x6 dot operator
    jacobian_i_pose = -Ks @ pi_derivative @ oTi @ end_term_pose # 4x6 Jacobian for i
    jacobian[4*i:4*i+4, -6:] = jacobian_i_pose

  # 4Nt x 4Nt invert term
  invert_term = jacobian @ joint_sigma @ jacobian.T + cross_copy(V, lpos.shape[0])
  k_gain = joint_sigma @ jacobian.T @ np.linalg.inv(invert_term) # 3M+6 x 4Nt 
  innovation = (observed_image_pos.T.reshape(-1) -
                predicted_image_pos.T.reshape(-1)) # 4Nt vector
  change = k_gain @ innovation # 3M+6 map/pose coordinate changes

  T_mu_update = T_mu @ expm(twist_hat(change[-6:-3],change[-3:]))
  mp_update = change[:threeM].reshape((mp.shape[0], 3)) # add this change
  joint_sigma_update = (np.eye(joint_sigma.shape[0]) - k_gain @ jacobian) @ joint_sigma

  return T_mu_update, mp_update, joint_sigma_update


def plot_map(mp, pose):
  # Plot the IMU pose and final landmark positions. The path starts from (0,0)
  landmarks = mp[mp[:,3]>0,:]
  plt.scatter(landmarks[:,0], landmarks[:,1], c='black', s=1)
  x_p = [pp[:,3] for pp in pose]
  x_p = np.array(x_p)
  x_p = x_p / (x_p[:,3][:, None])
  plt.plot(x_p[:,0], x_p[:,1])





if __name__ == '__main__':

  # Load the measurements
  filename = "./data/10.npz"
  t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

  # Sample 1 of every 10 features
  features = features[:,::20,:]

  # opticalTimu
  oTi = inv_se(imu_T_cam)

  Ks = np.zeros((4,4))
  Ks[:2,:3] = K[:2,:3]
  Ks[2:,:3] = K[:2,:3]
  Ks[2,3] = -K[0,0]*b

  M = features.shape[1]
  mp = np.full((M, 4), -1, dtype='float64') # Initial landmark pos
  mp_id = np.arange(M)

  # Landmark only covariance matrix for Part B
  mp_sig = np.eye(3*M)

  # Initial SE(3) pose, IMU is inverted
  T_mu = np.eye(4)
  T_mu[1,1] = -1
  T_mu[2,2] = -1

  # Initial joint covariance matrix with map and pose
  joint_sigma = np.eye(3*M + 6)

  # history of pose for plotting
  pose = [T_mu.copy()]

  for it in tqdm(np.arange(0,t.shape[1])):
    if(it>0): # start predicting from second time step
      tau = t[0,it] - t[0,it-1]
      v = linear_velocity[:,it] # Linear velocity for the tau timestep
      w = angular_velocity[:,it]
      # Part A IMU Localization via EKF Prediction
      T_mu = joint_se_estimate(T_mu,tau,v,w,joint_sigma,M)
      pose.append(T_mu.copy())

    valid_mask = features[0,:,it] >= 0 #observed features for current timestamp
    val_mp = mp_id[valid_mask] # Get indices of landmarks seen
    new_mask = (mp[val_mp,3] == -1)
    new_lmarks = val_mp[new_mask]
    upd_lmarks = val_mp[~new_mask]
    # Initialize unseen landmarks
    init_map_landmark(new_lmarks, it, mp, joint_sigma, features, T_mu, K, Ks, imu_T_cam, oTi)

    # Update pose and landmarks being seen again
    if upd_lmarks.size:
      # Map only update for Part B
      # update_map_landmark_individually(upd_lmarks, it, mp, mp_sig, features, T_mu, K, Ks, imu_T_cam, oTi)

      # Joint Update Part C
      T_mu_update, mp_update, joint_sigma_update = joint_update_map_pose(
          upd_lmarks, it, mp, joint_sigma, features, T_mu, K, Ks, imu_T_cam, oTi)
  
      mp[:,:3] = mp[:,:3] + mp_update
      joint_sigma = joint_sigma_update 
      T_mu = T_mu_update
      pose.append(T_mu.copy())

plot_map(mp, pose)
plt.show()

# Prediction only plot
# pose = np.array(pose)
# pose = pose.transpose((1,2,0))
# fig,ax = visualize_trajectory_2d(pose,path_name="Unknown",show_ori=True)





