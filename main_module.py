import cv2  #import opencv library
import cv2.aruco as aruco  #import aruco marker library
import numpy as np  #import numpy library
import RPi.GPIO as GPIO  #import module to control GPIOs on raspberry pi
import sys, time, os, math  #import system, time, operating system and math modules
import utils  #import utils module we created for image processing

#-- Setup GPIO pins
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)  #use board numbering
RW_PIN = 18;  #assign pin number where LW
LW_PIN = 13;
RW_ENA = 16;
LW_ENA = 11;
GPIO.setup(RW_PIN,GPIO.OUT) #set right wheel PWM pin as output
GPIO.setup(RW_ENA,GPIO.OUT) #set right wheel enable pin as output
GPIO.output(RW_ENA, 1)  #enable right wheel
GPIO.setup(LW_PIN,GPIO.OUT) #set left wheel PWM pin as output
GPIO.setup(LW_ENA,GPIO.OUT) #set left wheel enable pin as output
GPIO.output(LW_ENA, 1)  #enable left wheel

#-- initialize PWM
r = GPIO.PWM(RW_PIN,50) #initialize right wheel pin with arguments pin and frequency
r.start(0) #start right wheel PWM with initial duty cycle as 0
l = GPIO.PWM(LW_PIN,50) #initialize right wheel pin with arguments pin and frequency
l.start(0) #start left wheel PWM with initial duty cycle as 0

curveList = []  #create an empty array to store the curve values
avgVal = 10  #assign a value of 10 to be used to detect overflow

#-- Function to get the curve value
#-- Inputs: image(video input frame by frame), display selection (0 - no display, 1 - only resulting image with curve value, 2 - images at different steps stacked)
#-- Output: a signed integer curve value, a display of choice
def getLaneCurve(img, display):
   img = cv2.resize(img, (1280, 720))  #resize the frames to the same size used for calibration
   imgCopy = img.copy()  # 
   imgResult = img.copy()
   
   #-- step 1: thresholding frames
   imgThres = utils.thresholding(img)  #input original image frames to thresholding function in utils module

   #-- step 2: warping frames
   hT, wT, c = img.shape  #get image width and height (channels not used)
   points = utils.valTrackbars()  #get lane reference points from current trackbar(valTrackbars() is a function we created to place points on four lane edges)
   imgWarp = utils.warpImg(imgThres, points, wT,hT)  #get warped frame by inputting thresholded frame, lane reference points, width and height
   imgWarpPoints = utils.drawPoints(imgCopy, points)  #draw 

   #-- step 3: 
   midPoint, imgHist =  utils.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
   curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.9)
   curveRaw = curveAveragePoint-midPoint

   ### step 4
   curveList.append(curveRaw)
   if len(curveList)>avgVal:
       curveList.pop(0)
   curve = int(sum(curveList)/len(curveList))


   ### final step
   if display != 0:
       imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inv=True)
       imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
       imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
       imgLaneColor = np.zeros_like(img)
       imgLaneColor[:] = 0, 255, 0
       imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
       imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
       midY = 450
       cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
       cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
       cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
       for x in range(-30, 30):
           w = wT // 20
           cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                    (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
       #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
       #cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
   if display == 2:
       imgStacked = utils.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                            [imgHist, imgLaneColor, imgResult]))
       cv2.imshow('ImageStack', imgStacked)
   elif display == 1:
       cv2.imshow('Resutlt', imgResult)

   #--- Define Tag
   id_to_find  = 0  #id of the aruco marker to be found is 0
   marker_size  = 4 #marker size in centimeters is 4

   #--- Get the camera calibration path
   calib_path  = "/home/pi/mu_code/SeniorDesign/Aruco_pose_est/"

   #-- Load camera matrix and camera distortion
   camera_matrix   = np.load(calib_path+'camera_matrix.npy')
   camera_distortion   = np.load(calib_path+'camera_distortion.npy')

   #--- 180 deg rotation matrix around the x axis
   R_flip  = np.zeros((3,3), dtype=np.float32)
   R_flip[0,0] = 1.0
   R_flip[1,1] =-1.0
   R_flip[2,2] =-1.0

   #--- Define the aruco dictionary
   aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
   parameters  = aruco.DetectorParameters_create()

   #-- Convert in gray scale
   gray    = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #remember, OpenCV stores color images in Blue, Green, Red

   #-- Find all the aruco markers in the image
   corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)  #, cameraMatrix=camera_matrix, distCoeff=camera_distortion)

   #-- If an aruco marker is found check if it is a marker with id 0
   if ids is not None and ids[0] == id_to_find:
       ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)  #
       rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
       distance = tvec[2]
       current_velocity = 0  #initial velocity
       distance = round(distance, 2)

       if distance < 1:
         current_velocity_right = 0
         current_velocity_left = 0
       else if curve < 0:
         current_velocity_right = distance * .8
         current_velocity_left = (distance * .8) + curve
       else if curve > 0:
         current_velocity_right = distance * .8 + curve 
         current_velocity_left = distance * .8
       else:
         current_velocity_right = distance * .8
         current_velocity_left = distance * .8  
  
       r.ChangeDutyCycle(current_velocity_right)
       l.ChangeDutyCycle(current_velocity_left)

   else if (curve < 0):
        print('SLOW LEFT WHEEL!!!')
        r.ChangeDutyCycle(15)
        l.ChangeDutyCycle(12)
   else if (curve > 0):
        print('SLOW RIGHT WHEEL!!!')
        r.ChangeDutyCycle(12)
        l.ChangeDutyCycle(15)
   else:
        r.ChangeDutyCycle(20)
        l.ChangeDutyCycle(20)


if __name__ == '__main__':
   cap = cv2.VideoCapture(0)
   intialTrackBarVals = [102,80,20,214]
   utils.initializeTrackbars(intialTrackBarVals)
   frameCounter = 0
   while True:
       frameCounter += 1
       if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
           cap.set(cv2.CAP_PROP_POP_FRAMES, 0)
           frameCounter = 0
       success, img = cap.read()
       img = cv2.flip(img,-1)
       getLaneCurve(img, display = 1)
       cv2.waitKey(1)
