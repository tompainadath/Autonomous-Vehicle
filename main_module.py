mport cv2.aruco as aruco
import cv2
import numpy as np
import utils
import time
import os
import RPi.GPIO as GPIO
import sys, time, math
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
RW_PIN = 18;
LW_PIN = 13;
RW_ENA = 16;
LW_ENA = 11;
GPIO.setup(RW_PIN,GPIO.OUT) # Right Wheel
GPIO.setup(RW_ENA,GPIO.OUT) # Right Wheel
GPIO.output(RW_ENA, 1)
GPIO.setup(LW_PIN,GPIO.OUT) # Left Wheel
GPIO.setup(LW_ENA,GPIO.OUT) # Left Wheel
GPIO.output(LW_ENA, 1)

#initialize PWM
r = GPIO.PWM(RW_PIN,50) # Arguments are pin and frequency
r.start(0) # Argument is initial duty cycle, it should be 0
l = GPIO.PWM(LW_PIN,50) # Arguments are pin and frequency
l.start(0) # Argument is initial duty cycle, it should be 0

curveList = []
avgVal = 10
def getLaneCurve(img, display):
   img = cv2.resize(img, (480, 240))
   img2 = cv2.resize(img, (1280, 720))
   imgCopy = img.copy()
   imgResult = img.copy()
   ### step1
   imgThres = utils.thresholding(img)

   ### step 2
   hT, wT, c = img.shape
   points = utils.valTrackbars()
   imgWarp = utils.warpImg(imgThres, points, wT,hT)
   imgWarpPoints = utils.drawPoints(imgCopy, points)

   ### step 3
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
   id_to_find  = 0
   marker_size  = 4 #- [cm]


   def isRotationMatrix(R):
       Rt = np.transpose(R)
       shouldBeIdentity = np.dot(Rt, R)
       I = np.identity(3, dtype=R.dtype)
       n = np.linalg.norm(I - shouldBeIdentity)
       return n < 1e-6



   #--- Get the camera calibration path
   calib_path  = "/home/pi/mu_code/SeniorDesign/Aruco_pose_est/"

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
   gray    = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red

   #-- Find all the aruco markers in the image
   corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)#, cameraMatrix=camera_matrix, distCoeff=camera_distortion)

   if ids is not None and ids[0] == id_to_find:
       ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
       rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
       distance = tvec[2]
       v_f = 0
       v_t = 0
       lamda = 1
       start_time = time.process_time()
       end_time = time.process_time()
       elapsed_time = end_time - start_time
       distance = round(distance, 20)

       v_f = distance  # Vehicle velocity coefficient x Distance * need code for measuring distance *
       exp = math.exp(-lamda * elapsed_time)
       v_t = (v_f - v_t / (1 + exp))
       if v_t > 20:  #  Need to define max duty cycle for specific vehicle
           v_t = 20  # *Max duty cycle for specific vehicle*
           print('v_t_1', v_t)
           r.ChangeDutyCycle(v_t)
           l.ChangeDutyCycle(v_t)
       elif v_t < 5:
           v_t = 0
           r.ChangeDutyCycle(0)
           l.ChangeDutyCycle(0)
           print('v_t_2', v_t)
       else:
           print('v_t_3', v_t)
           r.ChangeDutyCycle(v_t)
           l.ChangeDutyCycle(v_t)
           if v_t > v_f:
               exp = math.exp(lamda * elapsed_time)
               v_t = (v_t - v_f / (1 + exp))
               print('v_t_4', v_t)
               if (v_t < 0):
                   print('v_t_5', v_t)
                   r.ChangeDutyCycle(0)
                   l.ChangeDutyCycle(0)
               else:
                   print('v_t_6', v_t)
                   r.ChangeDutyCycle(v_t)
                   l.ChangeDutyCycle(v_t)
       time.sleep(.01)
   if (curve < -15):
        print('STOP LEFT WHEEL!!!')
        r.ChangeDutyCycle(12)
        l.ChangeDutyCycle(15)
   else:
        r.ChangeDutyCycle(20)

   if (curve > 0):
        print('STOP RIGHT WHEEL!!!')
        l.ChangeDutyCycle(0)
   else:
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
