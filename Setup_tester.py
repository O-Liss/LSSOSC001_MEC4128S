import cv2
import numpy as np 
import tifffile as tf
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os

tif = tf.imread("/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Thermal/Sequence_breaking_20190726_1500_000000_297356.TIF")
img = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /bathandwa_502.jpg')
image = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/ice_test.jpg', cv2.IMREAD_GRAYSCALE)

tif_k = tif + 273.15
max_temp_k = np.amax(tif_k)
min_temp_k = np.amin(tif_k)
print(max_temp_k, min_temp_k)

# y=mx+c
m = (255-0)/(max_temp_k-min_temp_k)
c = -m*min_temp_k
tif_8bit = np.multiply(tif_k, m) + c
tif_8bit = np.uint8(tif_8bit)

cv2.imshow('first tiff in OpenCV', tif_8bit)

# perspective transform

# Pixel values in original image
red_point = [350, 75]
green_point = [600, 75]
black_point = [100, 503]
blue_point = [800, 503]

# Create point matrix
point_matrix = np.float32([red_point, green_point, black_point, blue_point])

# Draw lines to combine the points

cv2.line(img, (350, 75), (600, 75), (0, 0, 255), 2)
cv2.line(img, (100, 503), (350, 75), (0, 0, 255), 2)
cv2.line(img, (800, 503), (100, 503), (0, 0, 255), 2)
cv2.line(img, (800, 503), (600, 75), (0, 0, 255), 2)

# Output image size
width, height = 500, 600

# Desired points value in output images
converted_red_pixel_value = [0, 0]
converted_green_pixel_value = [width, 0]
converted_black_pixel_value = [0, height]
converted_blue_pixel_value = [width, height]

# Convert points
converted_points = np.float32([converted_red_pixel_value, converted_green_pixel_value,
                               converted_black_pixel_value, converted_blue_pixel_value])

# perspective transform
perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)
image1_trans = cv2.warpPerspective(img, perspective_transform, (width, height))
cv2.imwrite(str("images/" + "pp_v/image_trans.jpg"), image1_trans)
cv2.imshow("Original Image", img)
cv2.imshow("transformed Image", image1_trans)
cv2.waitKey(0)

"""
Input: Sea-ice image. Start algorithm: 
1: GVF ← GVF derived from grayscale of input image. 
2: ICE ← binary ice image by the K-means method. 
3: LIGHT ← binary “light” ice image by the thresholding method.
4: DARK ← ICE − LIGHT. 
5: SL ← Seeds of LIGHT found by local maxima of dis- tance transform.
6: SD ← Seeds of DARK found by local maxima of dis- tance transform.
7: for each seed sl ∈ SL do 
8:rl ← local maxima values at sl. 
9: icl ← initial contours locate at sl with its radius rl.
10: BL ← boundary detected by performing the snake algo- rithm based on GVF and icl.
11: LIGHT ← LIGHT with BL superimposed. 
12: end for 
13: for each seed sd ∈ SD do 
rd ← local maxima values at sd.
15:icd ← initial contours locate at sd with its radius rd.
16: BD ← boundary detected by performing the snake- algorithm based on GVF and icd.
17: DARK ← DARK with BD superimposed. 18: end for 19: SEGMENTATION ← LIGHT +DARK (LIGHT and DARK are marked differently in PIECES).
20: return SEGMENTATION. Output: Sea-ice segmentation.
C
"""

#contours of sea ice image using k-means using 3 clusters (light ice, dark ice, open water) using open CV package
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# number of clusters (K)
k = 3
pixel_values = np.float32(tif_8bit.reshape(-1, 1))
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#create light ice image for the k-means.
# create a mask to extract the light ice form the tif_8bit image and create new binary image
mask = np.zeros(tif_8bit.shape, dtype=np.uint8)
# convert labels into same dimensions as tif_8bit
labels = labels.reshape(tif_8bit.shape)
# set light ice pixels to 1 and everything else to 0 in the mask
mask[labels == 0] = 255
cv2.imshow('light ice', mask)


#do the same for the dark ice
mask2 = np.zeros(tif_8bit.shape, dtype=np.uint8)
# convert labels into same dimensions as tif_8bit
labels = labels.reshape(tif_8bit.shape)
# set dark ice pixels to 1 and everything else to 0 in the mask
mask2[labels == 1] = 180
cv2.imshow('dark ice', mask2)

mask3 = mask+mask2
cv2.imshow('all ice', mask3)


# Taking a matrix of size 1 as the kernel
kernel = np.ones((3, 3), np.uint8)

# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
img_erosion = cv2.erode(mask3, kernel, iterations=1)
img_dilation = cv2.dilate(mask3, kernel, iterations=1)

eros_dila = cv2.dilate(img_erosion, kernel, iterations=1)

cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.imshow('both', eros_dila)

cv2.waitKey(0)





