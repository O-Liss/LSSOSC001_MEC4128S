import cv2
import os
import tifffile as tf
import numpy as np

# Directory containing your images
image_dir = ('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Thermal/open water')

# Output video file name
output_video = 'output_video_thermal.mp4'

# Frame rate (you can adjust this)
frame_rate = 5

# Function to get the list of image files in the directory
def get_image_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', 'bmp', '.tif', '.tiff'))]

# Get the list of image files
image_files = get_image_files(image_dir)

# Sort the image files in alphanumeric order
image_files.sort()

# Get the first image to extract its dimensions
# first_image = cv2.imread(image_files[0])
# height, width, layers = first_image.shape

# Get the first image to extract its dimensions
first_image = tf.imread(image_files[0])
height, width = first_image.shape
print(height, width)
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height), isColor=False)

# Loop through the image files and add them to the video
for image_file in image_files:
    tif = tf.imread(image_file)

    tif_k = tif + 273.15
    max_temp_k = np.amax(tif_k)
    min_temp_k = np.amin(tif_k)
    # print(max_temp_k, min_temp_k)

    # y=mx+c
    m = (255 - 0) / (max_temp_k - min_temp_k)
    c = -m * min_temp_k
    tif_8bit = np.multiply(tif_k, m) + c
    tif_8bit = np.uint8(tif_8bit)

    # Invert the binary image
    inverted_image = 255 - tif_8bit

    # Resize the frame to match the video dimensions (width, height)
    resized_frame = cv2.resize(inverted_image, (width, height))

    # frame = cv2.imread(image_file)
    video.write(resized_frame)

# Release the VideoWriter object and close any open windows
video.release()
cv2.destroyAllWindows()

print(f'Video saved as {output_video}')