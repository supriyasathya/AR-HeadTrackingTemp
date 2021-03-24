# Code adapted from LearnOpenCV.com 
# This code is to test head pose estimation using OpenCV library. The image frames are read from Intel Realsense D400.

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
skip_frame = 3

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
    #image points are the landmark fiducial points on the image plane corresponding to the 3D model points (model_pts)
    image_pts = np.float32([shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]])
    #OpenCV solvepnp function to compute head pose
    (success, rotation_vec, translation_vec) = cv2.solvePnP(model_pts, image_pts, cam_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)#, flags=cv2.CV_ITERATIVE)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    R_rvec = R.from_rotvec(rotation_vec.transpose())
    R_rotmat = R_rvec.as_matrix()
    print(rotation_mat, R_rotmat)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, rotation_vec, rotation_mat, translation_vec, euler_angle, image_pts


def main():
# Set up the Realsense streams
# Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
# Intrinsics & Extrinsics of the Realsense camera stream
    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    mtx = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])
    dist = np.array(intr.coeffs)
    
    #load cascade classifier training file for lbpcascade 
    lbp_face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")#"/home/supriya/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml") #cv2.CascadeClassifier('/home/supriya/supriya/FSA-Net/demo/lbpcascade_frontalface.xml')   #cv2.CascadeClassifier('data/lbpcascade_frontalface_improved.xml') # cv2.CascadeClassifier('/home/supriya/supriya/FSA-Net/demo/lbpcascade_frontalface.xml')  
    #OpenCV model for landmark detection
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel('./lbfmodel.yaml')
     
    print('Start detecting pose ...')

    # img_idx keeps track of image index (frame #).
    img_idx = 0
    while True:
        # get video frame
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
  
        if not aligned_depth_frame or not color_frame:
            continue
        # read color frame
        input_img = np.asanyarray(color_frame.get_data())
        #increment count of the image index 
        img_idx = img_idx + 1
        
        # Process the first frame and every frame after "skip_frame" frames
        if img_idx == 1 or img_idx % skip_frame == 0:
            # convert image to grayscale
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            # detect faces using LBP detector
            faces = lbp_face_cascade.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5)
            #draw rectangle around detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            depth_point = [0,0,0]
            # if a face a detected, proceed with pose detection, etc.
            if len(faces) > 0:
               #detect landmarks
               status, landmarks = facemark.fit(gray_img, faces)
               #draw dots on the detected facial landmarks
               for f in range(len(landmarks)):
                   cv2.face.drawFacemarks(input_img, landmarks[f])
               #get head pose
               reprojectdst, rotation_vec, rotation_mat, translation_vec, euler_angle, image_pts = get_head_pose(landmarks[0][0], mtx, dist)
               # draw circle on image points (nose tip, corner of eye, lip and chin)
               for p in image_pts:
                   cv2.circle(input_img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
               
               # draw line sticking out of the nose tip and showing the head pose               
               (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vec, translation_vec, mtx, dist)
               p1 = ( int(image_pts[0][0]), int(image_pts[0][1]))
               p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
               cv2.line(input_img, p1, p2, (255,0,0), 2)

               #get 3D co-ord of nose - to get a more accurate estimation of the translaion of the head
               depth = aligned_depth_frame.get_distance(landmarks[0][0][30][0],landmarks[0][0][30][1] )
               depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [landmarks[0][0][30][0],landmarks[0][0][30][1]], depth)
               depth_point = np.array(depth_point)
               depth_point = np.reshape(depth_point,[1,3])
               
               # print Euler angles on image
               cv2.putText(input_img, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
               cv2.putText(input_img, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
               cv2.putText(input_img, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

         
            else:
               print("No Face Detected")
            # display detected face with landmarks, pose.   
            cv2.imshow('Landmark_Window', input_img)
                    
           
        key = cv2.waitKey(1)


if __name__ == '__main__':
    main()


