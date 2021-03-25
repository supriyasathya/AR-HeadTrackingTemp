OneTimeCalibration.py:
Main code to perform one-time calibration between Realsense and Magicleap

arucocalibclass.py  
An instance of arucocalibclass is called if single Aruco marker is used as Marker 1. 

charucocalibclass.py
An instance of charucocalibclass is called if a charuco board is used instead of a single Aruco marker

verifyOTB.py
use this code if the one-time calibration is already done and you have the pose of the marker 2 with respect to realsense and you can track an aruco marker or Charuco board and stream the pose to Magicleap.

computetrackingerror.py
script to compute error or change in pose between two transformation matrices. In the accuracy measurement experiments, A1 is the pose of a cube before and A2 is the pose after any manual adjustment of the virtual rendering to align with the real-world counterpart.

charuco_pose.py
script to check if charuco board pose if being detected correctly when using Realsense.

arucodetectRSML.py
this code can be used to perform calibration using aruco marker/charuco board and stream the transformed pose to Magicleap (without doing one-time calibration).
The startTracking function of this code is called by the Onetimecalibration code. 

