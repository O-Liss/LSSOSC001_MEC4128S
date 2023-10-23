# Description: This script is used for live capture and analysis from the camera
from SEA_ICE_Functions import *
import os

# Use calibration.py to manually set parameters
pixel_area_ratio = 1/700
threshold = 127
factor = 10
x_shift = 500
y_shift = 500

# Open the camera you want to use
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize frame counter and output directory
frame_counter = 0
output_directory = "frame_data"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly and it's time to save a frame
    if ret and frame_counter % 150 == 0:  # 150 frames * 1/30 seconds per frame = 5 seconds
        # Save the frame with a unique filename
        frame_filename = os.path.join(output_directory, f"frame_{frame_counter/150}.jpg")
        cv2.imwrite(frame_filename, frame)
        # Analysis the sea ice in the frame
        img, image_trans = analysis(frame, threshold, factor, x_shift, y_shift, frame_counter/150, pixel_area_ratio)
        # Concatenate 'img' and 'image_trans' horizontally
        combined_frame = cv2.hconcat([img, image_trans])
        # Save the frame with a unique filename
        out_filename = os.path.join(output_directory, f"out_frame_{frame_counter / 150}.jpg")
        cv2.imwrite(out_filename, combined_frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

    frame_counter += 1

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()