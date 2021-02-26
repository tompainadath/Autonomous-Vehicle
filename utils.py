import cv2
import numpy as np

#-- Function to threshold frame
def thresholding(img):
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert color to grayscale
    retval2, threshold = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #threshold grayscaled frame using OTSU method
    return threshold  #return thresholded frame

#-- Function to warp frame
def warpImg (img,points,w,h,inv=False):
   pts1 = np.float32(points)  #assign actual points to variable
   pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])  #assign desired points to variable
   if inv:
       matrix = cv2.getPerspectiveTransform(pts2,pts1)
   else:
       matrix = cv2.getPerspectiveTransform(pts1,pts2)
   imgWarp = cv2.warpPerspective(img,matrix,(w,h))  #warp the frame 
   return imgWarp  #return warped frame

#-- a dummy function
def nothing(a):
   pass

#-- Function to create a trackbar window and track bars
def initializeTrackbars(intialTracbarVals,wT=480, hT=240):
   cv2.namedWindow("Trackbars")  #name the trackbar window
   cv2.resizeWindow("Trackbars", 360, 240)  #define the size of the window
   cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)  #create top width trackbar
   cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)  #create top height trackbar
   cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)  #create bottom width trackbar
   cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)  #create bottom height trackbar

#-- Function to acquire trackbar values
def valTrackbars(wT=480, hT=240):
   widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")  #get top width trackbar value
   heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")  #get top height trackbar value
   widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")  #get bottom width trackbar value
   heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")  #get bottom height trackbar value
   points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                     (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])  #create an array of numbers using the above tackbar values
   return points  #return the points array
   
#-- Function to draw the points on the frame
def drawPoints(img,points):
   for x in range(4):
       cv2.circle(img,(int(points[x,0]),int(points[x][1])), 15, (0,0,255),cv2.FILLED)  #put a blob on the points
   return img  #retrun the fram with points drawn

#-- Function get histogram of a frame
def getHistogram(img, maxPer = 0.1, display = False, region=1):
   if region ==1:
       histValues = np.sum(img, axis=0)  #add up all the pixels in a frame column by column
   else:
       histValues = np.sum(img[img.shape[0]//region:,:], axis=0)  #add up only the pixels column by column within the specified number of rows
   minValue = np.min(histValues)  #find the minimum value from the frame 
   maxValue = maxPer*minValue  #find the maximum value with allowed deviation percentage 

   indexArray = np.where(histValues == maxValue)  #find the index of histogram values equal to the maximum value

   basePoint = int(np.average(indexArray))  #find the average of the indexes with maximum value and assign it to basepoint value

   if display:  #if diplay parameter is set to True
       imgHist = np.zeros((img.shape[0], img.shape[1],3), np.uint8)  #get a new array with zeros in the shape
       for x,intensity in enumerate(histValues):
           cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255//region), (255,0,255), 1)  #draw the shape
           cv2.circle(imgHist, (basePoint, img.shape[0]),20, (0,255,255), cv2.FILLED)  #put a blob at the basepoint index
           return basePoint,imgHist  #return base point and histogram
   return basePoint  #return basep point

#-- Function to visualize the process of lane detection
def stackImages(scale,imgArray):
   rows = len(imgArray)
   cols = len(imgArray[0])
   rowsAvailable = isinstance(imgArray[0], list)
   width = imgArray[0][0].shape[1]
   height = imgArray[0][0].shape[0]
   if rowsAvailable:
       for x in range ( 0, rows):
           for y in range(0, cols):
               if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                   imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
               else:
                   imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
               if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
       imageBlank = np.zeros((height, width, 3), np.uint8)
       hor = [imageBlank]*rows
       hor_con = [imageBlank]*rows
       for x in range(0, rows):
           hor[x] = np.hstack(imgArray[x])
       ver = np.vstack(hor)
   else:
       for x in range(0, rows):
           if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
               imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
           else:
               imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
           if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
       hor= np.hstack(imgArray)
       ver = hor
   return ver
