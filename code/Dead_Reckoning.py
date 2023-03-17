#!/usr/bin/env python
# coding: utf-8

# In[188]:


#### IMPORTING NECESSARY MODULES ####

from load_data import *
from pr2_utils import *
import matplotlib.pyplot as plt
import numpy as np
from numba import njit,prange
from numba import jit
import cv2
from tqdm import tqdm
import transforms3d


# In[ ]:


### DEFINING THE NECESSARY VARIABLES ###

dataset = 20
N = 100
lidar_data = np.load("/home/prithwiraj/Desktop/ECE276A_PR2/data/Hokuyo%d.npz"%dataset)
encoder_data = np.load("/home/prithwiraj/Desktop/ECE276A_PR2/data/Encoders%d.npz"%dataset)
imu_data=np.load("/home/prithwiraj/Desktop/ECE276A_PR2/data/Imu%d.npz"%dataset)
wheel_base = 0.311



# In[189]:


np.set_printoptions(threshold=np.inf)

lidar_range_data= lidar_data["ranges"]
lidar_angle_inc=lidar_data["angle_increment"]
lidar_stamps = lidar_data["time_stamps"]
lidar_angle_min = lidar_data["angle_min"] # start angle of the scan [rad]
lidar_angle_max = lidar_data["angle_max"] # end angle of the scan [rad]
lidar_angles=np.linspace(lidar_angle_min,lidar_angle_max,1081)


# In[191]:


def lidar_2_cart(r_data):
    '''
    converts lidar range data to cartesian coordinates in lidar frame
    input -> r_data : lidar scan in range

    '''
    lidar_angles=np.linspace(lidar_angle_min,lidar_angle_max,1081)
    X_L=r_data*np.cos(lidar_angles)
    Y_L=r_data*np.sin(lidar_angles)
    Z_L=[0]*1081
    cart=np.vstack((X_L,Y_L,Z_L))
    return cart.T

def lidar_cart_2_body(p):
    X,Y,Z=p
    R_lidar_2_body = np.eye(3)
    lidar_2_body = np.matmul(R_lidar_2_body,np.array([[X],[Y],[Z]]))+np.array([[0.13673],[0],[0.320675]])
    X_B,Y_B,Z_B=lidar_2_body
    return [X_B,Y_B,Z_B]


lidar_cart=np.apply_along_axis(lidar_2_cart,0,lidar_range_data)

lidar_mes_body=np.empty(list(lidar_cart.shape))

for i in range(lidar_mes_body.shape[2]):
    lidar_mes_body[:,:,i]=np.apply_along_axis(lidar_cart_2_body,1,lidar_cart[:,:,i])[:,:,0]


# In[192]:


## ENCODER READINGS ##

encoder_counts = encoder_data["counts"] # 4 x n encoder counts
encoder_stamps = encoder_data["time_stamps"] # encoder time stamps

print(encoder_stamps.shape)

dist_RW=(encoder_counts[0,:] + encoder_counts[2,:])*0.0022/2
dist_LW=(encoder_counts[1,:] + encoder_counts[3,:])*0.0022/2

encoder_tau=np.empty([1,encoder_stamps.shape[0]-1])

for i in range(encoder_stamps.shape[0]-1):
    encoder_tau[0,i]=encoder_stamps[i+1]-encoder_stamps[i]

V_R=dist_RW[1:]/encoder_tau[0,:]
V_L=dist_LW[1:]/encoder_tau[0,:]

V_Robot=(V_R+V_L)/2
yaw_rate = (V_R - V_L)/wheel_base


# In[193]:


# Motion Model for Differential Drive Robot

imu_stamps = imu_data["time_stamps"]  # acquisition times of the imu measurements
imu_angular_velocity = imu_data["angular_velocity"] # angular velocity in rad/sec
imu_linear_acceleration = imu_data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)


# storing the timestamps

time_index = {}
for t in encoder_stamps:
  time_diff = np.abs(imu_stamps - t)
  index = np.argmin(time_diff)
  time_index[t] = index

time_index_lidar = {}
for t in encoder_stamps:
  time_diff = np.abs(lidar_stamps - t)
  index = np.argmin(time_diff)
  time_index_lidar[t] = index


tau=np.empty([1,encoder_stamps.shape[0]-1])
for i in range(encoder_stamps.shape[0]-1):
    tau[0][i]=encoder_stamps[i+1]-encoder_stamps[i]



# In[194]:


def lidar_scan_to_world(scan,pose):
    '''
    function to convert the lidar scan in body frame to world frame
    '''

    x,y,theta=pose
    #print(x,y,theta)
    Rot_mat = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    p = np.array([[x],[y],[0]])
    scan_world = np.matmul(Rot_mat,scan.T) + p
    return scan_world


# In[195]:


def update_map_reck(scan,map,pose):
    '''
    function updates the 2D occupancy grid map
    inputs -> scan : lidar scan in world frame, map : stored 2D occupancy map, pose: pose of the robot

    '''
   
    X_start=pose[0]
    Y_start=pose[1]
    x_rob_idx = 600 - X_start//0.05
    y_rob_idx = 600 - Y_start//0.05 

    X_scan_idx= 600 - scan[0,:]//0.05
    Y_scan_idx= 600 - scan[1,:]//0.05 

    p_occ = 0.8  # Probability that a cell is occupied given a laser scan reading
    p_free = 0.2  # Probability that a cell is free given a laser scan reading

    log_odds_r = np.log(p_occ/p_free)
    log_odds_max = 4 * log_odds_r
    log_odds_min = -4 * log_odds_r


    for i,j in zip(X_scan_idx,Y_scan_idx):
        free_cells = bresenham2D(x_rob_idx, y_rob_idx, i, j).astype(int)
        x_free=free_cells[0,:-1]
        y_free=free_cells[1,:-1]
        map[x_free,y_free] -= log_odds_r
        map[int(i),int(j)] += log_odds_r
    
    map[np.where(map > log_odds_max)] = log_odds_max
    map[np.where(map < log_odds_min)] = log_odds_min
    
    return map


# In[196]:


# MAP INITIALIZARION WITH DEAD RECKONING

X_traj=np.zeros([3,encoder_stamps.shape[0]])

test_grid_map = np.zeros([1000,1000],dtype=np.float16)
lidar_scan_in_body_init = lidar_mes_body[:,:,time_index_lidar[encoder_stamps[0]]]
lidar_scan_init = lidar_scan_to_world(lidar_scan_in_body_init,X_traj[:,0])
test_grid_map = update_map_reck(lidar_scan_init,test_grid_map,X_traj[:,0])


# In[197]:


log_odds_val = np.empty([1,encoder_stamps.shape[0]-1])

map_pmf = np.exp(test_grid_map)/(1+np.exp(test_grid_map))

for t in tqdm(range(encoder_stamps.shape[0]-1)):

  V_Rob = V_Robot[t]
  X_traj[0,t+1] = X_traj[0,t] + tau[0,t] *  V_Rob * np.cos(X_traj[2,t])
  X_traj[1,t+1] = X_traj[1,t] + tau[0,t] *  V_Rob * np.sin(X_traj[2,t])
  X_traj[2,t+1] = X_traj[2,t] + tau[0,t] *  imu_angular_velocity[2,time_index[encoder_stamps[t]]]

  lidar_scan_in_body = lidar_mes_body[:,:,time_index_lidar[encoder_stamps[t+1]]] # using the next lidar scan for mapping
  lidar_scan_world = lidar_scan_to_world(lidar_scan_in_body,X_traj[:,t+1])
  test_grid_map = update_map_reck(lidar_scan_world,test_grid_map,X_traj[:,t+1])



# In[198]:


import pickle
with open('DR_traj_Dataset{}.pickle'.format(dataset), 'wb') as f:
    pickle.dump(X_traj, f)

plt.plot(X_traj[0,:],X_traj[1,:])
plt.title('Dead Reckoning Trajectory for Dataset {}'.format(dataset))
plt.xlabel('X in world frame')
plt.ylabel('Y in world frame')
plt.savefig('Dead Reckoning Trajectory for Dataset {}'.format(dataset))
plt.show()


# In[199]:


import pickle
with open('DR_map_Dataset{}.pickle'.format(dataset), 'wb') as f:
    pickle.dump(test_grid_map, f)

plt.imshow(test_grid_map, cmap='gray')
plt.title('Occupancy Map with Dead Reckoning Trajectory for Dataset {}'.format(dataset))
plt.savefig('Occupancy Map with Dead Reckoning Trajectory for Dataset {}'.format(dataset))
plt.show()


# TEXTURE MAPPING USING DEAD RECKONING TRAJECTORY

# In[200]:


import numpy as np
kinect_data = np.load("/home/prithwiraj/Desktop/ECE276A_PR2/data/Kinect%d.npz"%dataset)
disp_stamps = kinect_data["disparity_time_stamps"] # acquisition times of the disparity images
rgb_stamps = kinect_data["rgb_time_stamps"] # acquisition times of the rgb images

print(disp_stamps[0])
print(rgb_stamps[0])

time_index_rgb_disp={}
for t in rgb_stamps:
    time_diff = np.abs(disp_stamps - t)
    index = np.argmin(time_diff)
    time_index_rgb_disp[t]=index

time_index_rgb_encoder={}
for t in rgb_stamps:
    time_diff = np.abs(encoder_stamps - t)
    index = np.argmin(time_diff)
    time_index_rgb_encoder[t]=index

time_index_encoder_rgb={}
for t in encoder_stamps:
    time_diff = np.abs(rgb_stamps - t)
    index = np.argmin(time_diff)
    time_index_encoder_rgb[t]=index
 


# In[201]:


plt.plot(X_traj[2,:])
plt.title('Dead Reckoning Yaw angle for Dataset {}'.format(dataset))
plt.xlabel('Timestamps')
plt.ylabel('Angle (radians)')
plt.savefig('Dead Reckoning Yaw angle for Dataset {}'.format(dataset))
plt.show()


# In[205]:


def pix_cam_to_body(image):
    '''
    transforms from pixel locations camera regular frame to robot body frame
    '''
    pitch = 0.36 #rad
    yaw = 0.021 #rad
    roll=0.0
    pos = np.array([[0.18],[0.005],[0.36]])
    Rot_y = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]]) # pitch rotation
    Rot_z = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]) # yaw rotation
    Rot_mat =transforms3d.euler.euler2mat(roll,pitch,yaw)
    res = np.matmul(Rot_mat,image) + pos
    return res

def pixel_to_cam(image):         

    '''
    transforms pixel locations to camera regular frame
    '''
    K = np.array([[585.05108211, 0, 242.94140713],[0, 585.05108211, 315.83800193],[0, 0, 1.0]])
   
    reg_to_opt = np.array([[0,-1,0],[0,0,-1],[1,0,0]])

    opt_loc = np.matmul(np.linalg.inv(K),image)
    reg_loc = np.matmul(np.linalg.inv(reg_to_opt),opt_loc)

    return reg_loc

def pix_body_to_world(image,pose):
    '''
    function to convert the pixel locations in body frame to world frame
    '''
    x,y,theta=pose
    Rot_mat = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    p1 = np.array([[x],[y],[0.254]])
    pix_world = np.matmul(Rot_mat,image) + p1
    return pix_world

def color_map(p):

    '''
    Assigns RGB values to the Map
    '''
    global color_grid

    x,y,z,r,g,b = p # world frame coordinates of corresponding pixel
    

    x_cell_idx = (1000 - x//0.027).astype(int)
    y_cell_idx = (1000 - y//0.027).astype(int)
    
    if np.abs(z) <= 0.15:
      
        color_grid[x_cell_idx,y_cell_idx,:] = np.array([r,g,b],dtype='uint')
     


# In[210]:


color_grid = np.zeros([1000,1000,3],dtype='uint8')


disp_path = "/home/prithwiraj/Desktop/ECE276A_PR2/data/dataRGBD/Disparity20/"
rgb_path = "/home/prithwiraj/Desktop/ECE276A_PR2/data/dataRGBD/RGB20/"

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)


for t in range(rgb_stamps.shape[0]-1): 
  disp_idx = time_index_rgb_disp[rgb_stamps[t]]+1
  
  imd = cv2.imread(disp_path+'disparity{}_{}.png'.format(20,disp_idx),cv2.IMREAD_UNCHANGED) # (480 x 640)
  imc = cv2.imread(rgb_path+'rgb{}_{}.png'.format(20, t+1))[...,::-1] # (480 x 640 x 3)
  r, g, b = cv2.split(imc)

  # convert from disparity from uint16 to double
  disparity = imd.astype(np.float32)

  # get depth
  dd = (-0.00304 * disparity + 3.31)
  z = 1.03 / dd


  # calculate u and v coordinates 
  v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
  #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

  # get 3D coordinates 
  fx = 585.05108211
  fy = 585.05108211
  cx = 315.83800193
  cy = 242.94140713
  x = (u-cx) / fx * z
  y = (v-cy) / fy * z

  # calculate the location of each pixel in the RGB image
  rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
  rgbv = np.round((v * 526.37 + 16662.0)/fy)
  valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])

  rows = rgbu.shape[0]
  cols = rgbv.shape[1]

  rgbu = rgbu.reshape(-1)
  rgbv = rgbv.reshape(-1)
  z=z.reshape(-1)
  u = rgbu * z
  v = rgbv * z
  image_mat = np.vstack((u,v,z))

  cam_frame_coords = pixel_to_cam(image_mat)
  body_frame_coords = pix_cam_to_body(cam_frame_coords)
  world_frame_coords = pix_body_to_world(body_frame_coords,X_traj[:,time_index_rgb_encoder[rgb_stamps[t]]])

  X_world = world_frame_coords[0,:]
  Y_world = world_frame_coords[1,:]
  Z_world = world_frame_coords[2,:]

  ind_thres = np.where(Z_world <=0.5)
  Y_world = Y_world[ind_thres]
  X_world = X_world[ind_thres]


  x_cell_idx = (600 - (X_world//0.05)).astype(int)
  y_cell_idx = (600 - (Y_world//0.05)).astype(int)

  r = r.reshape([rows*cols,1])[ind_thres]
  g = g.reshape([rows*cols,1])[ind_thres]
  b = b.reshape([rows*cols,1])[ind_thres]

  color_grid[x_cell_idx,y_cell_idx,:] = np.hstack([r,g,b]) 

    
  

  

  


# In[211]:


import pickle
with open('Texture_Map_DR_Dataset{}.pickle'.format(dataset), 'wb') as f:
    pickle.dump(color_grid, f)

plt.imshow(color_grid)
plt.title('Dead Reckoning Texture Map for Dataset {}'.format(dataset))
plt.savefig('Dead Reckoning Texture Map for Dataset {}'.format(dataset))
plt.show()

