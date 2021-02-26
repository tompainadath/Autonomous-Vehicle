from picamera import PiCamera
from time import sleep
import numpy as np
import cv2
import glob

#-- termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#-- Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)*21.43125 # 21.43125mm square size

#-- Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

camera = PiCamera()
camera.resolution = (1280, 720)  #specify resolution to calibrate the camera with

#-- load 50 images
for i in range(50):
    camera.start_preview()
    sleep(5)
    camera.capture('image{0:04d}.jpg'.format(i))
    camera.stop_preview()

images = glob.glob('/home/pi/mu_code/*.jpg')  #specify where to find the images

image_no = 1
for fname in images:
   img = cv2.imread(fname)
   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

   #-- Find the chess board corners
   ret, corners = cv2.findChessboardCorners(gray, (7,5),None)

   #-- If found, add object points, image points (after refining them)
   if ret == True:
       objpoints.append(objp)
       corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
       imgpoints.append(corners2)

       #-- Draw and display the corners
       img = cv2.drawChessboardCorners(img, (7,5), corners2,ret)
       cv2.imshow('img',img)
       cv2.waitKey(500)
       cv2.imwrite('/home/pi/mu_code/SeniorDesign/'+str(image_no)+'.jpg', img)
       image_no += 1
       
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(mtx)
print(dist)
