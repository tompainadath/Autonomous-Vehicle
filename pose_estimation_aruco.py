import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math

#--- Define Tag
id_to_find  = 0  #marker id to look for
marker_size  = 4  #size of each side of the marker [cm]

#--- Get the camera calibration path
calib_path  = "/home/pi/mu_code/SeniorDesign/Aruco_pose_est/"  #assign calibration path to a variable
camera_matrix   = np.load(calib_path+'camera_matrix.npy')  #assign camera matrix from numpy array file to a variable
camera_distortion   = np.load(calib_path+'camera_distortion.npy')  #assign camera distortion from numpy array file to a variable

#--- Define the aruco dictionary
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  #assign the correct Aruco dictionary to a variable
parameters  = aruco.DetectorParameters_create()  #assign the parameters used to detect marker to a variable


#--- Capture the videocamera (this may also be a video or a picture)
cap = cv2.VideoCapture(0)  #capture live video

#-- Set the camera size as the one it was calibrated with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  #set the frame width of the video to 1280. Note: this is the same width used in calibration
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  #set the frame height of the video to 720. Note: this is the same height used in calibration

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN  #assign style of font for text to be used to put text on output frames

#-- Runs forever until stopped once the program gets here
while True:
    #-- Read the camera frame
    ret, frame = cap.read()  #extract the frames from the video

    #-- Convert in gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #convert the frames from color to black and white. Note: OpenCV stores color images in Blue, Green, Red

    #-- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)  #detect the marker from the frames and return corners, id found and or not found data

    #-- Check if the id to find is found
    if ids is not None and ids[0] == id_to_find:
        
        #-- array of rotation and position of each marker in camera frame
        #-- ret = [rvec, tvec, ?]
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)  #estimate the pose and return the data

        #-- Unpack the output
        #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
        #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]  #assign the data to variables

        #-- Draw the detected marker and put a reference frame over it
        aruco.drawDetectedMarkers(frame, corners)  #draw corners around the found marker
        aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)  #draw the axis on the found the found marker

        #-- Print the tag position in camera frame
        str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])  #assign the string to put on the frame to a variable
        cv2.putText(frame, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)  #put the text on to the frame

        print("Distance measured: ", tvec[2])  #print the distance measured in the console

    #--- Display the frame
    cv2.imshow('frame', frame)  #display the frame along with the texts, axis and corners

    #--- use 'q' in keyboard to quit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
