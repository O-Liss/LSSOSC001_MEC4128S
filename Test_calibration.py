from SEA_ICE_Functions import *

def main():
    pixel_area_ratio = float(input("Enter the pixel area ratio of the test subject: "))

    # capture image from the camera
    # frame, frame_path = capture_image()

    # Alternatively, use a downloaded picture
    frame_path = "/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Calibration testing/floe_d260mm.jpg"

    # Perspective transform
    image_trans, factor, x, y, width, height = initial_perspective_transform(frame_path)

    # Set threshold
    threshold = set_threshold()

    # Convert to grayscale
    imgray = cv2.cvtColor(image_trans, cv2.COLOR_BGR2GRAY)

    # Find threshold
    ret, thresh = cv2.threshold(imgray, threshold, 4, 0)  # detect only the specified range

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Initialize variables to keep track of the largest contour and its area
    largest_contour = None
    largest_area = 0

    # Iterate through the contours to find the largest one
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    real_area = pixel_area_ratio * largest_area
    print("The area in m^2 of the subject is:", real_area)

    # Draw contours inliers
    cv2.drawContours(image_trans, largest_contour, -1, (0, 0, 255), 2)

    cv2.imshow('Test image', image_trans)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()