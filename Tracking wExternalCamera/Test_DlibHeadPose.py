import os
import cv2
import sys
sys.path.append('..')


import numpy as np
import cv2
print("OpenCV Version: {}".format(cv2.__version__))
from math import cos, sin
import cv2.aruco as aruco
import pyrealsense2 as rs

import dlib
from imutils import face_utils
import socket
import struct
from scipy.spatial.transform import Rotation as R
from arucocalibclass import arucocalibclass
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R

face_landmark_path = './shape_predictor_68_face_landmarks.dat'
skip_frame = 1

# 3D model points of the facial landmarks
model_pts = np.array([(0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape, cam_matrix, dist_coeffs):
    shape = face_utils.shape_to_np(shape)
    #image points are the landmark fiducial points on the image plane corresponding to the 3D model points (model_pts)
    image_pts = np.float32([shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]])
    #OpenCV solvepnp function to compute head pose
    (success, rotation_vec, translation_vec) = cv2.solvePnP(model_pts, image_pts, cam_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    R_rvec = R.from_rotvec(rotation_vec.transpose())
    R_rotmat = R_rvec.as_matrix()
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, rotation_vec, rotation_mat, translation_vec, euler_angle, image_pts


def main():
     # dlib face detector
    dlibdetector = dlib.get_frontal_face_detector()
     # dlib landmark detector
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
     
    print('Start detecting pose ...')
    detected_pre = []
    img_idx = 0 # img_idx keeps track of image index (frame #).

    while True:
        # get video frame
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        if not aligned_depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics of Realsense
        depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(
            color_frame.profile)
        mtx = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])
        dist = np.array(intr.coeffs)
        # grab frame and convert to np array
        input_img = np.asanyarray(color_frame.get_data())
        img_idx = img_idx + 1
        img_h, img_w, _ = np.shape(input_img)

        # Process the first frame and every frame after "skip_frame" frames
        if img_idx == 1 or img_idx % skip_frame == 0:

             
              gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
              #detect faces using dlib               
              rects = dlibdetector(gray_img, 1)
              input_imgcopy = input_img

              for rect in rects:
                 # detect facial landmarks
                  shape = predictor(gray_img, rect)    

                  #head pose estimation
                  reprojectdst, rotation_vec, rotation_mat, translation_vec, euler_angle, image_pts = get_head_pose(shape, mtx, dist)
                  
                  # Project a 3D point (0, 0, 1000.0) onto the image plane. We use this to draw a line sticking out of the nose
                  (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vec, translation_vec, mtx, dist)
                  for p in image_pts:
                       cv2.circle(input_img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                  # draw line sticking out of the nose tip and showing the head pose               
                  p1 = ( int(image_pts[0][0]), int(image_pts[0][1]))
                  p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                  cv2.line(input_img, p1, p2, (255,0,0), 2)
                  # convert landmarks detected to numpy type
                  shape = face_utils.shape_to_np(shape)                 
                  landmarks = np.float32(shape)
                  
                  # draw circle on facial landmarks
                  for (x, y) in landmarks:
                      cv2.circle(input_img, (x, y), 1, (0, 0, 255), -1)
                  #get 3D co-ord of nose 
                  depth = aligned_depth_frame.get_distance(image_pts[0][0], image_pts[0][1])
                  cv2.circle(input_img, (image_pts[0][0], image_pts[0][1]), 3, (0,255,0), -1)
                  depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [image_pts[0][0], image_pts[0][1]], depth)
                  depth_point = np.array(depth_point)
                  depth_point = np.reshape(depth_point,[1,3])    

                  
                  # print Euler angles on image
                  cv2.putText(input_img, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)
                  cv2.putText(input_img, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)
                  cv2.putText(input_img, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)
                 #Combine rotation matrix and translation vector (given by the depth point) to get the head pose 
                  RSTr = np.hstack([rotation_mat, depth_point.transpose()])
                  RSTr = np.vstack([RSTr,[0,0,0,1]])

                

        else:
               print("No Face Detected")
        cv2.imshow('Landmark_Window', input_img)
            
          
          
           
        key = cv2.waitKey(1)


if __name__ == '__main__':
    main()


