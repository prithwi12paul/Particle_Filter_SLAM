# Particle_Filter_SLAM


## Project Overview

This project aims to implement LiDAR-Based Particle Filter Simultaneous Localization and Mapping (SLAM) to estimate the pose of a differential-drive robot and
build a 2D-occupancy grid map of the environment. The careful fusion of sensor measurements such as odometry (encoder, IMU) and LiDAR are used to localize the robot and build an indoor 2-D map. Particle Filter SLAM is a popular method for estimating a robot's pose and creating a map of the unknown terrain it travels through. After the 2D occupancy grid map is built, the RGBD images of the Kinect sensor and the estimated robot trajectory are used to produce a 2D color map of the floor surface. 

The robot is equipped with encoder, IMU sensor, 2D Lidar and RGBD Stereo camera sensor where each sensor reports measurements at its own frequency hence the sensor data is not synchronized. Initially, the sensor data is synchronized by matching the corresponding timestamps and storing the data in a dictionary (mapping).

## Project File Structure

### Datasets

The [data](https://drive.google.com/file/d/14r2RIZEKrX5g59-mCGqjcHqhwIfc-3LH/view?usp=share_link) contains the dataset which has been collected by the graduate student researchers in the Existential Robotics Laboratory, University of California San Diego. The dataset contains the IMU measurements, RGBD pixel coordinates of the Kinect Sensor Stereo Camera, Encoder Measurements and the Hokyuo Lidar scans for a differential drive robot moving in an indoor environment.
### Source Code

#### Necessary Python Libraries

The third party modules used are as listed below. They are included as [`requirements.txt`](code/requirements.txt).

- ipython==7.31.1
- matplotlib==3.5.2
- numpy==1.21.5
- pandas==1.5.3
- transforms3d==0.4.1
  opencv-python==4.7.0.72


Python files

- [load_data.py](code/load_data.py) - Loads the .npz files
  [PF_utils_functions.py](code/PF_utils_functions.py) - Necessary user-defined Python functions to implement Particle Filter SLAM
- [Particle_Filter_SLAM.py](code/Particle_Filter_SLAM.py) - Main python file that runs the Particle Filter SLAM algorithm

### Jupyter Notebook

The [Jupyter Notebook](code/Particle_Filter_SLAM.ipynb) is the notebook version of the main python code which has all the visualization and plots for the results

## How to run the code

Install all required libraries -

```
pip install -r requirements.txt

```
Run the load_data.py file -

```
python3 load_data.py

```
Run the PF_utils_functions.py file
```

python3 PF_utils_functions.py
```

Dead Reckoning Trajectory-
- Open the [Dead_Reckoning.py](code/Dead_Reckoning.py) file,
- Specify the dataset number and wheelbase
- Specify the location path of the lidar_data, encoder_data, IMU_data, disparity and RGB images
- Run the python file

```
python3 Dead_Reckoning.py
```

- Run the Particle_Filter_SLAM.py file

```
python3 Particle_Filter_SLAM.py
```







