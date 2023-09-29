import cv2
import numpy as np

# Load the input grayscale image
input_image = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/235.jpg', cv2.IMREAD_GRAYSCALE)

# Define camera parameters
focal_length_mm = 2.0  # Focal length in millimeters
shooting_angle_deg = 30.0  # Shooting angle in degrees
field_of_view_deg = 46.0  # Field of view in degrees

# Image dimensions
image_width_pixels = input_image.shape[1]
image_height_pixels = input_image.shape[0]

# Define the trapezoid vertices (order: top-left, top-right, bottom-right, bottom-left)
trapezoid_vertices = np.array([
    [100, 100],  # Adjust these coordinates as needed
    [500, 100],  # Adjust these coordinates as needed
    [600, 600],  # Adjust these coordinates as needed
    [0, 600],    # Adjust these coordinates as needed
], dtype=np.float32)

# Calculate the perspective transformation matrix using cv2.getPerspectiveTransform
field_of_view_rad = np.radians(field_of_view_deg)
aspect_ratio = image_width_pixels / image_height_pixels
fx = focal_length_mm / image_width_pixels
fy = fx * aspect_ratio
cx = image_width_pixels / 2
cy = image_height_pixels / 2

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Output image size
width, height = 600,600

# Define the center of the image as the projection point
projection_point = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# Calculate the transformation matrix using cv2.getPerspectiveTransform
proj_matrix = cv2.getPerspectiveTransform(trapezoid_vertices, projection_point)

# Perform perspective transform using cv2.warpPerspective
output_image = cv2.warpPerspective(input_image, proj_matrix, (width, height))

# Draw the trapezoid on the original image
trapezoid_color = (0, 255, 0)  # Green color in BGR format
cv2.polylines(input_image, [trapezoid_vertices.astype(int)], isClosed=True, color=trapezoid_color, thickness=2)

# Display the original image with the trapezoid drawn
cv2.imshow('Original Image with Trapezoid', input_image)
cv2.imshow('Transformed Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# You can save the transformed image using cv2.imwrite if needed:
# cv2.imwrite('output_transformed_image.jpg', output_image)