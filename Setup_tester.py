import cv2
import numpy as np 
import tifffile as tf
import matplotlib as plt
import os

tif = tf.imread("/Users/oscarliss/Desktop/LSSOSC001_FINALYEARPROJECT/Sequence_breaking_20190726_1500_000000_297356.TIF")
img = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_FINALYEARPROJECT/thumbnail.color.png')
img2 = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_FINALYEARPROJECT/20197211418.67.tiff')


tif_k = tif + 273.15
maxtemp_k = np.amax(tif_k)
mintemp_k = np.amin(tif_k)
print(maxtemp_k, mintemp_k)

# y=mx+c
m = (255-0)/(maxtemp_k-mintemp_k)
c = -m*mintemp_k
tif_8bit = np.multiply(tif_k, m) + c
tif_8bit = np.uint8(tif_8bit)

cv2.imshow('first tiff in OpenCV', tif_8bit)

# Create a contour of the sea ice edges in thermal image
thresh = cv2.threshold(tif_8bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Draw contours onto thermal image
for cntr in contours:
    cv2.drawContours(tif_8bit, [cntr], 0, (0, 0, 255), 2)

# Display image

cv2.imshow('contours', tif_8bit)


# converting image into grayscale image
# imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# cv2.imshow('thermal images', img)
# cv2.imshow('visual images', img2)
# cv2.imshow('grey scale images', imgGry)
cv2.waitKey(0)
