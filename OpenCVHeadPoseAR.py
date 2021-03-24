# 
import os
import cv2
import sys
sys.path.append('..')
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import dlib
from imutils import face_utils
import socket
import struct
from scipy.spatial.transform import Rotation as R
from arucocalibclass import arucocalibclass
from numpy.linalg import inv

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
    (success, rotation_vec, translation_vec) = cv2.solvePnP(model_pts, image_pts, cam_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    R_rvec = R.from_rotvec(rotation_vec.transpose())
    R_rotmat = R_rvec.as_matrix()
    #print(rotation_mat, R_rotmat)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, rotation_vec, rotation_mat, translation_vec, euler_angle, image_pts


def simplified_calib(T_M2_RS, c):
    #T_M2_ML is the pose of marker 2 in AR headset space
    T_M2_ML = np.zeros([4, 4], np.float32)
    n_avg = 0
    N_samples = 2 #number of samples of T_M2_ML to be averaged out and taken.
    while (True):
        # get T_M2_ML from AR headset
        dataRecv = c.recv(64)
        MLArr = np.frombuffer(dataRecv, dtype=np.float32)
        print("verify MLArr", MLArr)
        if MLArr[3] != 0:
            T_M2_ML += MLArr.reshape(4, 4)
            print("T_M2_ML", T_M2_ML)
            n_avg += 1
            if n_avg == N_samples:
                T_M2_ML = T_M2_ML / N_samples
                print("T_M2_ML average", T_M2_ML)                
                # compute T_RS_ML2 (transformation from Realsense space to AR Headset space
                T_RS_ML = np.matmul(T_M2_ML, inv(T_M2_RS))
                break
    return T_RS_ML

def main():

    face_landmark_path = './shape_predictor_68_face_landmarks.dat'
    #flags/initializations to be set
    skip_frame = 3 
    socket_connect = 1 # set to 0 if we are testing the code locally on the computer with only the realsense tracking.
    simplified_calib_flag = 0 # this is set to 1 if we want to do one-time calibration
    arucoposeflag = 1
    # img_idx keeps track of image index (frame #).
    img_idx = 0
    N_samples_calib = 10 # number of samples for computing the calibration matrix using homography


    if socket_connect == 1:
    #  create a socket object
        s = socket.socket()
        print ("Socket successfully created")
    # reserve a port on your computer in our case it is 2020 but it can be anything
        port = 2020
        s.bind(('', port))
        print ("socket binded to %s" %(port))

       # put the socket into listening mode
        s.listen(5)
        print ("socket is listening")
        c,addr = s.accept()
        print('got connection from ',addr)
    
    if socket_connect == 1 and simplified_calib_flag == 0:
      arucoinstance = arucocalibclass()
      ReturnFlag = 1
      aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
      marker_len = 0.0645
      MLRSTr = arucoinstance.startcamerastreaming(c, ReturnFlag, marker_len, aruco_dict, N_samples_calib)
      print(MLRSTr)
    elif socket_connect == 1 and simplified_calib_flag == 1:
      T_M2_RS = np.array([-1.0001641  , 0.00756584  ,0.00479072 , 0.03984956,-0.00774137, -0.99988126 ,-0.03246199 ,-0.01359556,
            0.00453644, -0.03251681,  0.99971441 ,-0.00428408,  0.   ,       0.       ,   0.   ,       1.        ])
      T_M2_RS = T_M2_RS.reshape(4, 4)
      MLRSTr = simplified_calib(T_M2_RS, c)
    else:
      MLRSTr = np.array((1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1))
      MLRSTr = MLRSTr.reshape(4,4)
      print(MLRSTr)
   # end socket ######
 
# Configure depth and color streams of Realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

# Intrinsics & Extrinsics of Realsense
    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    mtx = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])
    dist = np.array(intr.coeffs)
    #load cascade classifier training file for lbpcascade 
    lbp_face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")#"/home/supriya/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml") #cv2.CascadeClassifier('/home/supriya/supriya/FSA-Net/demo/lbpcascade_frontalface.xml')   #cv2.CascadeClassifier('data/lbpcascade_frontalface_improved.xml') # cv2.CascadeClassifier('/home/supriya/supriya/FSA-Net/demo/lbpcascade_frontalface.xml')  
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel('./lbfmodel.yaml')
     
    print('Start detecting pose ...')
    detected_pre = []
    
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
        img_h, img_w, _ = np.shape(input_img)

        # Process the first frame and every frame after "skip_frame" frames
        if img_idx == 1 or img_idx % skip_frame == 0:
            # convert image to grayscale
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            # detect faces using LBP detector
            faces = lbp_face_cascade.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 5)
             #draw rectangle around detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            depth_point = [0,0,0]
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
              # draw circle on image points (nose tip, corner of eye, lip and chin)
               for p in image_pts:
                   cv2.circle(input_img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
               
               # draw line sticking out of the nose tip and showing the head pose               
               (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vec, translation_vec, mtx, dist)
               p1 = ( int(image_pts[0][0]), int(image_pts[0][1]))
               p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
               cv2.line(input_img, p1, p2, (255,0,0), 2)

               # get 3D co-ord of nose - to get a more accurate estimation of the translaion of the head
               depth = aligned_depth_frame.get_distance(landmarks[0][0][30][0],landmarks[0][0][30][1] )
               cv2.circle(input_img, (landmarks[0][0][30][0],landmarks[0][0][30][1]), 3, (0,255,0), -1)
               depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [landmarks[0][0][30][0],landmarks[0][0][30][1]], depth)
               depth_point = np.array(depth_point)
               depth_point = np.reshape(depth_point,[1,3])
               
               #check if the depth estimation is not zero and filters out faces within 0.8 m from the camera
               if (depth_point[0][2]!=0 and depth_point[0][2] < 0.8):#
                   #Combine rotation matrix and translation vector (given by the depth point) to get the head pose 
                   RSTr = np.hstack([rotation_mat, depth_point.transpose()])
                   RSTr = np.vstack([RSTr,[0,0,0,1]])
                   print("head pose", RSTr)
                  
                   # If arucoposeflag = 1, an Aruco marker will  get detected and its transformed pose will be streamed to the AR headset and the pose
                   # of the tracking parent will be updated to reflect Aruco marker pose. This can be used to verify/test the accuracy of teh calibration
                   if arucoposeflag == 1:
                             print("aruco")
                             gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    		                    # set dictionary size depending on the aruco marker selected
                             aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    		                    # detector parameters can be set here (List of detection parameters[3])
                             parameters = aruco.DetectorParameters_create()
                             parameters.adaptiveThreshConstant = 10
    		                    # lists of ids and the corners belonging to each id 
                             corners, ids, rejectedImgPoindetectorts = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    		                    # font for displaying text (below)
                             font = cv2.FONT_HERSHEY_SIMPLEX
    		                    # check if the ids list is not empty
                             if np.all(ids != None):
    		                    # estimate pose of each marker and return the values
                               rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.045, mtx, dist)# 0.0628 (0.061 if using Dell laptop - 95% zoom)
                               for i in range(0, ids.size):
    		                    # draw axis for the aruco markers
                                 aruco.drawAxis(input_img, mtx, dist, rvec[i], tvec[i], 0.1)

    		                    # draw a square around the markers
                               aruco.drawDetectedMarkers(input_img, corners)
                               #Combine rotation matrix and translation vector to get Aruco pose
                               R_rvec = R.from_rotvec(rvec[0])
                               R_rotmat = R_rvec.as_matrix()
                               AruRSTr = np.hstack([R_rotmat[0],tvec[0].transpose()])
                               AruRSTr = np.vstack([AruRSTr,[0,0,0,1]])
                               RSTr = AruRSTr
                               print("Aruco pose", AruRSTr)
                               
                   # Since pose detected in OpenCV will be right handed coordinate system, it needs to be converted to left-handed coordinate system of Unity
                   RSTr_LH = np.array([RSTr[0][0],RSTr[0][2],RSTr[0][1],RSTr[0][3],RSTr[2][0],RSTr[2][2],RSTr[2][1],RSTr[2][3],RSTr[1][0],RSTr[1][2],RSTr[1][1],RSTr[1][3],RSTr[3][0],RSTr[3][1],RSTr[3][2],RSTr[3][3]])# converting to left handed coordinate system
                   RSTr_LH = RSTr_LH.reshape(4,4)
                  #Compute the transformed pose to be streamed to MagicLeap               
                   HeadPoseTr = np.matmul(MLRSTr,RSTr_LH)
                   
                   if socket_connect == 1: 
                      # Array to be sent
                     ArrToSend = np.array([HeadPoseTr[0][0],HeadPoseTr[0][1],HeadPoseTr[0][2],HeadPoseTr[0][3],HeadPoseTr[1][0],HeadPoseTr[1][1],HeadPoseTr[1][2],HeadPoseTr[1][3],HeadPoseTr[2][0],HeadPoseTr[2][1],HeadPoseTr[2][2],HeadPoseTr[2][3],HeadPoseTr[3][0],HeadPoseTr[3][1],HeadPoseTr[3][2],HeadPoseTr[3][3]])
                      # Send transformed pose to AR headset
                     dataTosend = struct.pack('f'*len(ArrToSend),*ArrToSend)
                     c.send(dataTosend)
            else:
               print("No Face Detected")
            # display detected face with landmarks, pose.  
            cv2.imshow('Landmark_Window', input_img)

           
        key = cv2.waitKey(1)


if __name__ == '__main__':
    main()


