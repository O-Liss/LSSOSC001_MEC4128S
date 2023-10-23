import cv2
import numpy as np
from skimage.segmentation import active_contour

# Load the grayscale image
image = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/image_trans.jpg', cv2.IMREAD_GRAYSCALE)

# Perform K-means clustering to segment the image into 3 clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 5
pixel_values = np.float32(image.reshape(-1, 1))
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert labels to image shape
clustered_image = labels.reshape(image.shape).astype(np.uint8)

# Create a binary ice image where dark ice and light ice are 1, and open water is 0
binary_ice_image = np.zeros_like(clustered_image)
binary_ice_image[(clustered_image == 1) | (clustered_image == 2)] = 1
binary_ice_image2 = np.zeros_like(clustered_image)
binary_ice_image2[(clustered_image == 1) | (clustered_image == 2)] = 255

# Invert the binary ice image to measure distance from ice to open water
inverted_binary_ice_image = 1 - binary_ice_image

# Perform distance transform with "City Block" (L1) distance metric on the inverted binary ice image
distance_transform = cv2.distanceTransform(binary_ice_image, cv2.DIST_L1, 3, dstType=cv2.CV_32F)

cv2.imshow('distance', distance_transform)
# Define a threshold for merging seeds (Tseed)
Tseed = 4  # Adjust this threshold as needed


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

# Convert contours to the format suitable for GVF snake (list of lists of points)
snake_contours = [contour.tolist() for contour in contours]

# Create a list to hold the corrected contours
corrected_contours = []

# Convert each contour to the correct format
for contour_points in snake_contours:
    corrected_contours.append(np.array(contour_points, dtype=np.int32))

# Convert corrected_contours to a single NumPy array
snake_corrected_contours = np.vstack(corrected_contours)

# Perform active contour with GVF snake
snake = active_contour(image, snake_corrected_contours, alpha=0.1, beta=10, gamma=0.001, w_line=0, w_edge=300, max_num_iter=100, convergence=0.1, boundary_condition='periodic')

# Display the result with seeds, corrected contours, and active contour
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for seed in seeds:
    x, y, _ = seed
    cv2.circle(result_image, (x, y), 5, (0, 0, 255), -1)
for contour in corrected_contours:
    cv2.polylines(result_image, [contour], isClosed=True, color=(0, 255, 0), thickness=1)
cv2.polylines(result_image, [snake.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=1)

# display the number of contours on the image
cv2.putText(result_image, str("Number of seeds = " + str(len(corrected_contours))),(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0))

# Display the result
cv2.imshow('Result with Seeds and Contours', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

