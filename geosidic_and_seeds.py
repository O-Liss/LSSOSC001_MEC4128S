import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.color import rgb2gray
from skimage.util import img_as_float
import time

start_time = time.time()
# Load the grayscale image
image = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/image_trans.jpg', cv2.IMREAD_GRAYSCALE)

# Perform K-means clustering to segment the image into 3 clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3
pixel_values = np.float32(image.reshape(-1, 1))
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Show the clustered image
# Now convert back into uint8, and make original image
center = np.uint8(centers)
res = center[labels.flatten()]
res2 = res.reshape(image.shape)
#cv2.imshow('res2',res2)
#cv2.waitKey(0)

# Convert labels to image shape
clustered_image = labels.reshape(image.shape).astype(np.uint8)

# Create a binary ice image where dark ice and light ice are 1, and open water is 0
binary_ice_image = np.zeros_like(clustered_image)
binary_ice_image[(clustered_image == 1) | (clustered_image == 2)] = 1
binary_ice_image2 = np.zeros_like(clustered_image)
binary_ice_image2[(clustered_image == 1) | (clustered_image == 2)] = 255

#cv2.imshow('bin ice image', binary_ice_image2)
#cv2.waitKey(0)

# Invert the binary ice image to measure distance from ice to open water
inverted_binary_ice_image = 1 - binary_ice_image

# Perform distance transform with "City Block" (L1) distance metric on the inverted binary ice image
distance_transform = cv2.distanceTransform(inverted_binary_ice_image, cv2.DIST_L1, 3, dstType=cv2.CV_32F)

cv2.imshow('distance', distance_transform)
# Define a threshold for merging seeds (Tseed)
Tseed = 5  # Adjust this threshold as needed


# Find regional maxima and merge them
def merge_maxima(distance_transform, Tseed):
    maxima = np.zeros_like(distance_transform)
    h, w = distance_transform.shape

    for i in range(h):
        for j in range(w):
            if distance_transform[i, j] >= Tseed:
                maxima[i, j] = 1

    return maxima


merged_maxima = merge_maxima(distance_transform, Tseed)


# Find seeds (centers of regional maxima and merged regions)
def find_seeds(merged_maxima):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(merged_maxima.astype(np.uint8))
    seeds = []

    for i in range(1, len(stats)):  # Skip background label 0
        x, y = int(centroids[i][0]), int(centroids[i][1])
        value = distance_transform[y, x]
        seeds.append((x, y, value))

    return seeds


seeds = find_seeds(merged_maxima)

# Generate circular contours around each seed
contours = []
for seed in seeds:
    x, y, value = seed
    radius = value / np.sqrt(2)
    contour = []
    for angle in range(0, 360, 5):  # Adjust the step size as needed
        x_contour = int(x + radius * np.cos(np.radians(angle)))
        y_contour = int(y + radius * np.sin(np.radians(angle)))
        contour.append([x_contour, y_contour])
    contours.append(np.array(contour, dtype=np.int32))

# plot the contours on the original image
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')
for contour in contours:
    plt.plot(contour[:, 0], contour[:, 1], linewidth=2)
plt.show()



def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


# Load the grayscale image
im = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/image_trans.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocess the image to highlight contours
gimage = inverse_gaussian_gradient(im)

# Initial starting pionts for the snake
init_ls = np.zeros(im.shape, dtype=np.int8)

# Add the contours to the initial level set
for contour in contours:
    cv2.fillPoly(init_ls, pts=[contour], color=1)

# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)

# Morphological GAC, balloon
ls = morphological_geodesic_active_contour(gimage, 300, init_ls, smoothing=1, balloon=5, threshold=0.7, iter_callback=callback)

end_time = time.time()
print("Time taken:", end_time - start_time)

fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(im, cmap="gray")
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors='b')
ax[0].set_title("Morphological GAC segmentation", fontsize=12)

ax[1].imshow(ls, cmap="gray")
ax[1].set_axis_off()
contour = ax[1].contour(evolution[0], [0.5], colors='r')
contour.collections[0].set_label("Starting Contour")
contour = ax[1].contour(evolution[5], [0.5], colors='g')
contour.collections[0].set_label("Iteration 5")
contour = ax[1].contour(evolution[-1], [0.5], colors='b')
contour.collections[0].set_label("Last Iteration")
ax[1].legend(loc="upper right")
title = "Morphological GAC Curve evolution"
ax[1].set_title(title, fontsize=12)

plt.show()
