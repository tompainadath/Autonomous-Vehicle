import cv2
import numpy as np
import utils

curveList = []  #initialize array for curve values
avgVal = 10  #define the length of the array of curve value

#-- Function get curve value
def getLaneCurve(img, display = 2):
   imgCopy = img.copy()
   imgResult = img.copy()
   
   #-- Step1: Threshold each frame
   imgThres = utils.thresholding(img)

   #-- Step 2: Warp each frame
   hT, wT, c = img.shape
   points = utils.valTrackbars()
   imgWarp = utils.warpImg(imgThres, points, wT,hT)
   imgWarpPoints = utils.drawPoints(imgCopy, points)

   #-- Step 3: Get midpoint and curve average point to find raw curve value
   midPoint, imgHist =  utils.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
   curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.9)
   curveRaw = curveAveragePoint-midPoint

   #-- Step 4: Find curve value
   curveList.append(curveRaw)
   if len(curveList)>avgVal:
       curveList.pop(0)
   curve = int(sum(curveList)/len(curveList))

   #-- Display options: 
   #-- if display not 0, make the component necessary to diplay
   #-- if display = 1, show only final result frames with curve value
   #-- if display = 2, show the whole process of image processing by stacking different frames
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
   if display == 2:
       imgStacked = utils.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                            [imgHist, imgLaneColor, imgResult]))
       cv2.imshow('ImageStack', imgStacked)
   elif display == 1:
       cv2.imshow('Resutlt', imgResult)

#-- This starts the whole lane detection module
if __name__ == '__main__':
   cap = cv2.VideoCapture('vid1.mp4')
   intialTrackBarVals = [102,80,20,214] 
   utils.initializeTrackbars(intialTrackBarVals)
   frameCounter = 0

   while True:
       frameCounter += 1
       if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
           cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
           frameCounter = 0

       success, img = cap.read()
       img = cv2.resize(img, (480, 240))
       curve = getLaneCurve(img, display = 2)
       print(curve)
       cv2.waitKey(1)

