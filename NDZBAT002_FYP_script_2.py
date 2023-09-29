import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import IPython.display as ipd
from tqdm.notebook import tqdm
import subprocess

print("Start")

image1 = cv.imread('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /bathandwa_502.jpg')
cv.imshow("original",image1)                                                               #shows image
cv.waitKey(0)   #delay 

def rescaleFrame(frame,scale = 0.5):   
    width = int(frame.shape[1]*scale)    #1 is for width
    height =int(frame.shape[0]*scale)    #0 is for height
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) # resizes the frame to the dimensions in all the area

resized_image = rescaleFrame(image1)   #resizing image
#cv.imshow("resized_pic",resized_image)  #showing resized image

# perspective transform

# Pixel values in original image                 
red_point = [350,75]
green_point = [600,75]
black_point = [100,503]
blue_point = [800,503]
 
# Create point matrix
point_matrix = np.float32([red_point,green_point,black_point, blue_point])
 
# Draw lines to combine the points                     

cv.line(image1, (350,75), (600,75), (0,0,255), 2) 
cv.line(image1, (100,503), (350,75), (0,0,255), 2)
cv.line(image1, (800,503), (100,503), (0,0,255), 2)
cv.line(image1, (800,503), (600,75), (0,0,255), 2)


# Output image size
width, height = 500,600
 
# Desired points value in output images    
converted_red_pixel_value = [0,0]
converted_green_pixel_value = [width,0]
converted_black_pixel_value = [0,height]
converted_blue_pixel_value = [width,height]
 
# Convert points
converted_points = np.float32([converted_red_pixel_value,converted_green_pixel_value,
                               converted_black_pixel_value,converted_blue_pixel_value])
 
# perspective transform
perspective_transform = cv.getPerspectiveTransform(point_matrix,converted_points)
image1_trans = cv.warpPerspective(image1,perspective_transform,(width,height))
cv.imwrite(str("image_trans.jpg"), image1_trans)
cv.imshow("Original Image", image1)
cv.imshow("transformed Image", image1_trans)
cv.waitKey(0)
con_area=0
def nothing(x):
    print(x)
    #print("con_area=",sum(con_area))
    #print("min_c=",min_c)
# Create a black image, a window
width, height = 500,600
cv.namedWindow('image')

#cv.createTrackbar('CP', 'image', 1, 255, nothing)
#cv.createTrackbar('G', 'image', 0, 255, nothing)
#cv.createTrackbar('R', 'image', 0, 255, nothing)
cv.createTrackbar("threshold1","image",127,255,nothing)
#cv.createTrackbar("threshold2","image",3,255,nothing)
switch = 'color/gray'
cv.createTrackbar(switch, 'image', 0, 1, nothing)

while(1):
    img = cv.imread("image_trans.jpg") #np.zeros((300,512,3), np.uint8) #
    imgray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    #cv.imshow('image',img)
    #pos= cv.getTrackbarPos('CP', 'image')
    threshold1 =cv.getTrackbarPos("threshold1","image")
    #threshold2 = cv.getTrackbarPos("threshold2","image")    
    font = cv.FONT_HERSHEY_SIMPLEX
    
    ret, thresh = cv.threshold(imgray, threshold1, 4,0)                      #detect only the specified range
    
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) #Only retrieves the extrem outer contours 
    #print("Number of contours = " + str(len(contours)))
    #print(contours[0])
    #cv.drawContours(image1_trans, contours, -1, (0, 255, 0), 1)
    con_area = []
    for j in contours:
        area = cv.contourArea(j)
    
        if area>200:
            con_area.append(area/700) 
    #min_c = min(con_area)
   
    cv.drawContours(img, contours, -1, (255,0,0),1)
  
    cv.putText(img,str("Number of contours = " + str(len(con_area))),(50,100),font,0.8,(0,0,255))
    cv.putText(img,str("Ice concentration in % ="+str(round(100*(sum(con_area))/((img.shape[0])*img.shape[1]/700),2))),(50,130),font,0.8,(0,0,255)) 
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        print("done with K")
        break

    #b = cv.getTrackbarPos('B', 'image')
    #g = cv.getTrackbarPos('G', 'image')
    #r = cv.getTrackbarPos('R', 'image')
    s = cv.getTrackbarPos(switch, 'image')

    if s == 0:
        pass 
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.imshow('image',img)


cv.destroyAllWindows()
print("con max",max(con_area))
plt.hist(con_area,40,ec="red")
plt.xlabel("areas of Pancake Ice [x10 m^2]") 
plt.ylabel("number of pancakes")
plt.title("pancake distribution ")  #with total contour area of "+str(sum(con_area))
print("Done")


# read in video
cap = cv.VideoCapture('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video01.mp4')

totalframecount= int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print("The total number of frames in this video is ", totalframecount)

frameTime = 1000 # time of each frame in ms, you can add logic to change this value.


while(cap.isOpened()):
    ret, frame = cap.read()
    cv.imshow('frame',frame)
    if cv.waitKey(frameTime) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()






