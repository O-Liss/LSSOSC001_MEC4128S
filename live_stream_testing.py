import numpy as np
import cv2 as cv
import os

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

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
        cv.imwrite(frame_filename, frame)
        print(f"Saved frame {frame_counter} as {frame_filename}")

    # Display the resulting frame
    cv.imshow('frame', frame)

    # If 'q' is pressed, exit the loop
    if cv.waitKey(1) == ord('q'):
        break

    frame_counter += 1

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()