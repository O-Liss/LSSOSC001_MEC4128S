import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import logging
import pandas as pd
import math
import statistics

# Configure logging to write to a log file
logging.basicConfig(
    level=logging.INFO,
    filename='sea_ice_data.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_meta_data(videofile):
    """
    Get metadata from video file
    :param videofile: path of video file
    :return: number of frames, height of video, width of video, frames per second
    """

    # Capture video from file
    cap = cv2.VideoCapture(videofile)

    # Total number of frames in the video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video height and width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # frame rate (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    return n_frames, height, width, fps

def get_first_frame(videofile):
    """
    Get first frame from video file
    :param videofile: path of video file
    :return: the first frame of the video
    """

    video_cap = cv2.VideoCapture(videofile)
    success, image = video_cap.read()
    if success:
        cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG file

    return image

def nothing(x):
    """
    This function is used for the trackbar so that it can keep updating in the while loop
    """
    pass

def initial_perspective_transform(imagefile):
    """
    This function is used to set the perspective transform of the first frame of the video
    :param imagefile: path of the first frame of the video
    :return: transformed image, scaling factor, x shift number of pixels, y shift number of pixels, width of transformed image, height of transformed image
    """

    # Read in image
    image = cv2.imread(imagefile)

    # Sizing in original image
    h, w, c = image.shape

    x = int(w/2)
    y = int(h/5)
    length = 125

    # Sizing off the trapazoid based of the camera angle above open ocean
    # The back horizontal line are what the other lines are based off
    orth_x = 14/5
    orth_y = 214/125

    # Create a black image, a window
    # Output image size
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

    # Create a trackbar for scale
    cv2.createTrackbar("scale", "image", 1, 10, nothing)
    # Create a trackbar for x position
    cv2.createTrackbar("x shift", "image", 100, 1000, nothing)
    # Create a trackbar for y position
    cv2.createTrackbar("y shift", "image", 50, 500, nothing)

    while 1:
        # Read in image
        img = cv2.imread(imagefile)

        # Get trackbar positions
        factor = cv2.getTrackbarPos("scale", "image")
        x = cv2.getTrackbarPos("x shift", "image")
        y = cv2.getTrackbarPos("y shift", "image")

        # Pixel values in original image
        T_L = [(x - (length * factor) / 2), y]
        T_R = [(x + (length * factor) / 2), y]
        B_L = [(x - (length * factor * orth_x) / 2), (y + (length * factor * orth_y))]
        B_R = [(x + (length * factor * orth_x) / 2), (y + (length * factor * orth_y))]

        # Create point matrix
        point_matrix = np.float32([T_L, T_R, B_L, B_R])

        # Draw lines to combine the points
        cv2.line(img, (int(T_L[0]), int(T_L[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)
        cv2.line(img, (int(B_L[0]), int(B_L[1])), (int(T_L[0]), int(T_L[1])), (0, 0, 255), 2)
        cv2.line(img, (int(B_R[0]), int(B_R[1])), (int(B_L[0]), int(B_L[1])), (0, 0, 255), 2)
        cv2.line(img, (int(B_R[0]), int(B_R[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)

        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            print("done with K")
            break
        img = cv2.imshow('image', img)


    # Output image size of transformed image
    width, height = int(length * factor * 2), int(length * factor * 12 / 5)

    # Desired points value in output images
    converted_T_L = [0, 0]
    converted_T_R = [width, 0]
    converted_B_L = [0, height]
    converted_B_R = [width, height]

    # Convert points
    converted_points = np.float32([converted_T_L, converted_T_R,
                                   converted_B_L, converted_B_R])

    # perspective transform
    perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)
    image_trans = cv2.warpPerspective(image, perspective_transform, (width, height))
    cv2.imwrite(str("image_trans.jpg"), image_trans)
    cv2.imshow("transformed Image", image_trans)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return image_trans, factor, x, y, width, height

def perspective_transform(image, factor, x, y):
    """
    This function is used execute the perspective transform on the rest of the frames of the video
    :param image: frame from the video
    :param factor: the scaling factor from the initial_perspective_transform function
    :param x: the x shift number of pixels from the initial_perspective_transform function
    :param y: the y shift number of pixels from the initial_perspective_transform function
    :return: transformed image
    """

    # Sizing in original image
    h, w, c = image.shape
    length = 125

    orth_x = 14 / 5
    orth_y = 214 / 125

    # Pixel values in original image
    T_L = [(x - (length * factor) / 2), y]
    T_R = [(x + (length * factor) / 2), y]
    B_L = [(x - (length * factor * orth_x) / 2), (y + (length * factor * orth_y))]
    B_R = [(x + (length * factor * orth_x) / 2), (y + (length * factor * orth_y))]


    # Output image size of transformed image
    width, height = int(length * factor * 2), int(length * factor * 12 / 5)

    # Create point matrix
    point_matrix = np.float32([T_L, T_R, B_L, B_R])

    # Desired points value in output images
    converted_T_L = [0, 0]
    converted_T_R = [width, 0]
    converted_B_L = [0, height]
    converted_B_R = [width, height]

    # Convert points
    converted_points = np.float32([converted_T_L, converted_T_R,
                                   converted_B_L, converted_B_R])

    # perspective transform
    perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)
    image_trans = cv2.warpPerspective(image, perspective_transform, (width, height))

    # Draw lines to combine the points after the transform has taken place
    cv2.line(image, (int(T_L[0]), int(T_L[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)
    cv2.line(image, (int(B_L[0]), int(B_L[1])), (int(T_L[0]), int(T_L[1])), (0, 0, 255), 2)
    cv2.line(image, (int(B_R[0]), int(B_R[1])), (int(B_L[0]), int(B_L[1])), (0, 0, 255), 2)
    cv2.line(image, (int(B_R[0]), int(B_R[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)

    return image_trans

def set_threshold():
    """
    This function is used to set the threshold for the ice concentration
    :return: the threshold value
    """

    # Create a black image, a window
    width, height = 500, 600
    cv2.namedWindow('image')

    # Create a trackbar for threshold
    cv2.createTrackbar("threshold", "image", 127, 255, nothing)

    while 1:
        # Read in image
        img = cv2.imread("image_trans.jpg")  # np.zeros((300,512,3), np.uint8) #

        # Convert to grayscale
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get trackbar position
        threshold = cv2.getTrackbarPos("threshold", "image")

        # Select the font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Find threshold
        ret, thresh = cv2.threshold(imgray, threshold, 4, 0)  # detect only the specified range

        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Only retrieves the extreme outer contours

        con_area = []
        for j in contours:
            area = cv2.contourArea(j)

            if area > 200:
                con_area.append(area / 700)

        # Draw contours
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

        cv2.putText(img, str("Number of contours = " + str(len(con_area))), (50, 100), font, 0.8, (0, 0, 255))
        cv2.putText(img, str("Ice concentration in % =" + str(round(100 * (sum(con_area)) / ((img.shape[0]) * img.shape[1] / 700), 2))), (50, 130), font, 0.8,(0, 0, 255))

        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            print("done with K")
            break
        img = cv2.imshow('image', img)

    cv2.destroyAllWindows()

    return threshold

def find_outliers(contours, data_area, sensitivity=5):
    """
    This function is used to remove the outliers from the contours
    :param contours: the contours
    :param data_area: the contour data
    :param sensitivity: the sensitivity of the outlier detection
    :return: the inliers, the outliers, the number of inliers, the number of outliers
    """

    # Calculates the mean, standard deviation and upper and lower boundaries for inliers.
    m = np.mean(data_area)
    st_dev = statistics.stdev(data_area)
    u_bound = m + sensitivity * st_dev
    l_bound = m - sensitivity * st_dev

    inliers_contours = []
    outliers_contours = []
    inliers_areas = []
    # Remove the upper and lower bound outliers form the data and contours append outliers contour to outliers list
    for i in range(len(data_area)):
        if (data_area[i] > u_bound) or (data_area[i] < l_bound):
            # Creates outliers list
            outliers_contours.append(contours[i])

        else:
            # Creates inliers list
            inliers_contours.append(contours[i])
            # Creates inliers area list
            inliers_areas.append(data_area[i])


    # Calculate the number of inliers and outliers
    num_inliers = len(inliers_contours)
    num_outliers = len(outliers_contours)

    return tuple(inliers_contours), tuple(outliers_contours), inliers_areas, num_inliers, num_outliers


def analysis(frame, threshold, factor, x_shift, y_shift, frame_no):
    """
    This function is used to analyse the frames of the video
    :param frame: the frame of the video
    :param threshold: the threshold value taken from the set_threshold function
    :param factor: the scaling factor taken from the initial_perspective_transform function
    :param x_shift: the x shift number of pixels taken from the initial_perspective_transform function
    :param y_shift: the y shift number of pixels taken from the initial_perspective_transform function
    :param frame_no: the number of the frame in the video
    :return: the video frame with outline of trapazoid and info written on it, the transformed image with contours drawn on it
    """

    # Perspective transform
    image_trans = perspective_transform(frame, factor, x_shift, y_shift)

    # Sizing of transformed image
    w, h, c = image_trans.shape

    # Sizing of original image
    height, width, _ = frame.shape

    # Select the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Convert to grayscale
    imgray = cv2.cvtColor(image_trans, cv2.COLOR_BGR2GRAY)

    # Find threshold
    ret, thresh = cv2.threshold(imgray, threshold, 4, 0)  # detect only the specified range

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    num_contours = len(contours)
    # Remove the outliers from the contours
    raw_contour_area = []
    adjusted_contour_area = []
    pixel_area_ratio = 700
    for j in contours:
        area = cv2.contourArea(j)
        raw_contour_area.append(area)
        adjusted_contour_area.append(area / pixel_area_ratio)

    # Find the outliers
    contours, outliers_contours, inliers_areas, num_inliers, num_outliers = find_outliers(contours, raw_contour_area)

    # Draw contours inliers
    cv2.drawContours(image_trans, contours, -1, (0, 255, 0), 1)
    # Draw contours outliers
    cv2.drawContours(image_trans, outliers_contours, -1, (0, 0, 255), 1)

    # Percentage of outliers
    percentage_outliers = round((num_outliers/num_contours)*100, 2)

    # Catarogise contours into pancake, brash and frazil
    pancake = 0
    brash = 0
    frazil = 0
    for k in range(len(inliers_areas)):
        area = inliers_areas[k]
        if area > 1400:
            pancake += 1
        elif area > 700:
            brash += 1
        elif area > 200:
            frazil += 1

    SIC = round(100 * (sum(raw_contour_area)/pixel_area_ratio) / ((image_trans.shape[0]) * image_trans.shape[1] / 700), 2)
    time = '2023-09-30 12:00:00'  # Replace with actual time from camera

    cv2.putText(frame, str("Number of contours = " + str(num_inliers)), (width-450, 100), font, 0.8, (0, 0, 255))
    cv2.putText(frame, str("Percentage of outliers = " + str(percentage_outliers)), (width - 450, 130), font, 0.8, (0, 0, 255))
    cv2.putText(frame, str("Ice concentration in % =" + str(SIC)), (width-450, 160), font,0.8, (0, 0, 255))

    log_message = f"Frame {frame_no}: Num_Contours={num_contours}, Percentage_outliers={percentage_outliers}, Concentration={SIC}, Pancake_no={pancake}, Brash_no={brash}, Frazil_no={frazil}, Time={time}"
    logging.info(log_message)

    return frame, image_trans


def create_video(videofile, threshold, factor, x_shift, y_shift, trans_width,  trans_height):
    """
    This function is used to create the video with the analysis done on it
    :param videofile: the path of the video file
    :param threshold: the threshold value taken from the set_threshold function
    :param factor: the scaling factor taken from the initial_perspective_transform function
    :param x_shift: the x shift number of pixels taken from the initial_perspective_transform function
    :param y_shift: the y shift number of pixels taken from the initial_perspective_transform function
    :param trans_width: the width of the transformed image taken from the initial_perspective_transform function
    :param trans_height: the height of the transformed image taken from the initial_perspective_transform function
    :return: the video with the analysis done on it
    """
    n_frames, height, width, fps = get_meta_data(videofile)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out1 = cv2.VideoWriter("out_test01.mp4", fourcc, 10, (width + int(trans_width * height / trans_height), height), True)
    cap1 = cv2.VideoCapture(videofile)
    for i in tqdm(range(n_frames), total=n_frames):
        ret, img = cap1.read()
        if ret == False:
            break

        # Analysis
        img, image_trans = analysis(img, threshold, factor, x_shift, y_shift, (i+1))
        image_trans = cv2.resize(image_trans, (int(trans_width * height / trans_height), height))

        # Concatenate 'img' and 'image_trans' horizontally
        combined_frame = cv2.hconcat([img, image_trans])

        # Write the concatenated frame to the output video file
        out1.write(combined_frame)

    out1.release()
    cap1.release()

def plotting():
    # Read the data from the log file using pandas
    # log_data = pd.read_csv('sea_ice_data.log', header=None, names=['Timestamp', 'Level', 'Data'])
    #
    # # Extract the relevant columns
    # log_data['Frame Number'] = log_data['Data'].str.extract(r'Frame (\d+):').astype(float)
    # log_data['Concentration'] = log_data['Data'].str.extract(r'Concentration=([\d.]+),').astype(float)

    log_lines = []
    frames = []
    concs = []

    # Open the log file for reading
    with open('sea_ice_data.log', 'r') as file:
        # Read each line and append it to the list
        for line in file:
            log_lines.append(line.strip())

    for line in log_lines:
        frame_start = line.find('Frame ') + 6
        frame_end = line.find(': Num_Contours')
        frame = float(line[frame_start:frame_end])
        frames.append(frame)
        conc_start = line.find("Concentration=") + 14
        conc_end = line.find(", Pancake_no")
        conc = float(line[conc_start:conc_end])
        concs.append(conc)


    # Plot the data using matplotlib
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(frames, concs, label='Sea Ice Concentration')
    plt.xlabel('Frame number')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title('Sea Ice Data Over Time')
    plt.show()
