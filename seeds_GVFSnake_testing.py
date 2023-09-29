import cv2
import numpy as np

# Load the grayscale image
image = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/ice_test.jpg', cv2.IMREAD_GRAYSCALE)

# Perform K-means clustering to segment the image into 3 clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3
pixel_values = np.float32(image.reshape(-1, 1))
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert labels to image shape
clustered_image = labels.reshape(image.shape).astype(np.uint8)

# Create a binary ice image where dark ice and light ice are 1, and open water is 0
binary_ice_image = np.zeros_like(clustered_image)
binary_ice_image[(clustered_image == 1) | (clustered_image == 2)] = 1
binary_ice_image2 = np.zeros_like(clustered_image)
binary_ice_image2[(clustered_image == 1) | (clustered_image == 2)] = 255

cv2.imshow('bin ice image', binary_ice_image2)

# Invert the binary ice image to measure distance from ice to open water
inverted_binary_ice_image = 1 - binary_ice_image

# Perform distance transform with "City Block" (L1) distance metric on the inverted binary ice image
distance_transform = cv2.distanceTransform(binary_ice_image, cv2.DIST_L1, 3, dstType=cv2.CV_32F)

cv2.imshow('distance', distance_transform)
# Define a threshold for merging seeds (Tseed)
Tseed = 6  # Adjust this threshold as needed


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

# Display the result with seeds and circular contours
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for seed in seeds:
    x, y, _ = seed
    cv2.circle(result_image, (x, y), 5, (0, 0, 255), -1)
for contour in contours:
    cv2.polylines(result_image, [contour], isClosed=True, color=(0, 255, 0), thickness=1)

# Display the result
cv2.imshow('Result with Seeds and Contours', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute the GVF (Gradient Vector Field)
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
gvf_x = gradient_x / (np.sqrt(gradient_x**2 + gradient_y**2) + 1e-6)
gvf_y = gradient_y / (np.sqrt(gradient_x**2 + gradient_y**2) + 1e-6)

# Snake parameters (adjust as needed)
alpha = 0.1  # Weight of external energy (GVF)
beta = 0.2  # Weight of internal energy (curvature)
gamma = 0.2  # Step size for snake evolution
iterations = 100  # Number of iterations

# Perform GVF snake
for snake_points in snake_contours:
    snake = np.array(snake_points, dtype=int)

    for _ in range(iterations):
        # Compute external energy (GVF field)
        external_energy_x = cv2.remap(gvf_x, snake[:, 0].astype(np.float32), snake[:, 1].astype(np.float32),
                                      interpolation=cv2.INTER_LINEAR)
        external_energy_y = cv2.remap(gvf_y, snake[:, 0].astype(np.float32), snake[:, 1].astype(np.float32),
                                      interpolation=cv2.INTER_LINEAR)

        external_energy = np.column_stack((external_energy_x, external_energy_y))

        # Simplify the snake contour to remove self-intersections
        epsilon = 0.02 * cv2.arcLength(snake, closed=True)
        snake = cv2.approxPolyDP(snake, epsilon, closed=True)

        # Compute convex hull
        hull = cv2.convexHull(snake, returnPoints=False)

        # Compute convexity defects
        if len(hull) > 2:
            defects = cv2.convexityDefects(snake, hull)
            if defects is not None:
                internal_energy = defects.squeeze()[:, 3]
            else:
                internal_energy = np.zeros(len(snake))
        else:
            internal_energy = np.zeros(len(snake))

        # Ensure internal_energy has the same number of rows as snake points
        internal_energy = np.resize(internal_energy, snake.shape[0])

        # Update snake points using the dot product
        updated_points = snake + gamma * (alpha * external_energy + np.dot(internal_energy, external_energy.T) * beta)
        # Round the updated points to integers (if needed)
        snake = np.round(updated_points).astype(int)

    # Draw the final contour on the original image
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in snake:
        cv2.circle(result_image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)

    # Display the result with the snake contour