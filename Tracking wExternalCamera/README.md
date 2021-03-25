Download the folder "TrackingwExternalCamera" to your Windows/Mac/Ubuntu.

The code was tested on Mac within a conda environment with Python 3.7. Following commands were used to create the environment:
conda create -n py_3_7 python=3.7
conda activate py_3_7

A conda environment is recommended.

List of libraries and packages to be installed are listed in requirements.txt.
For the Python wrapper for Realsense, you can use this command: pip install pyrealsense2
For Opencv-Python, you can use: pip3 install opencv-python and pip3 install opencv-contrib-python
For dlib-python, you can use: pip install dlib
For numpy, use: pip3 install numpy


Overview of the scripts in this repo:
Test_OpenCVHeadPose.py
# Test if Realsense camera is interfaced correctly and head pose estimation using OpenCV works correctly.

Test_DlibHeadPose.py
# Test if Realsense camera is interfaced correctly and head pose estimation using Dlib works correctly.

OpenCVHeadPoseAR.py
# Calibration and markerless head tracking on AR device using OpenCV Facemark for face detection and facial landmark detection. Head pose is computed using OpenCV solvePnP.

OpenCVHeadPoseAR_Mean.py
# Calibration and markerless head tracking on AR device using OpenCV Facemark for face detection and facial landmark detection. Head pose is computed using OpenCV solvePnP. Mean of the head pose every few frames is streamed to the AR device to make the tracking smoother than sending the pose every frame.

OpenCVHeadPoseARKalman.py
# Calibration and markerless head tracking on AR device using OpenCV Facemark for face detection and facial landmark detection. Head pose is computed using OpenCV solvePnP.
Includes Kalman filter for pose tracking.

DlibHeadPoseAR.py
# Calibration and markerless head tracking on AR device using Dlib for face detection and facial landmark detection. Head pose is computed using OpenCV solvePnP.

DlibHeadPoseARKalman.py
# Calibration and markerless head tracking on AR device using Dlib for face detection and facial landmark detection. Head pose is computed using OpenCV solvePnP.
Includes Kalman filter for pose tracking.



arucocalibclass.py  
# An instance of arucocalibclass is called if single Aruco marker is used as Marker 1. 

charucocalibclass.py
# An instance of charucocalibclass is called if a charuco board is used instead of a single Aruco marker


charuco_pose.py
# script to check if charuco board pose if being detected correctly when using Realsense.

computetrackingerror.py
# script to compute error or change in pose between two transformation matrices.



Marker-basedTrackingAR.py
# 1. aruco marker detection with Realsense camera (set socket_connect = 0 to test Realsense tracking Aruco pose only)
# 2. calibrate the realsense and AR headset spaces using marker 1, and test the calibration by tracking an Aruco marker (set socket_connet = 1). This can be used to
# perform marker-based head tracking by attaching an Aruco marker to subject's head and tracking. Details in the Google doc.

