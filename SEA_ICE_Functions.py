import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import logging
import math
import statistics
import time

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
    length = 25

    # Sizing off the trapazoid based of the camera angle above open ocean
    # The back horizontal line are what the other lines are based off
    orth_x = 10/5
    orth_y = 214/125

    # Create a black image, a window
    # Output image size
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

    # Create a trackbar for scale
    cv2.createTrackbar("scale", "image", 1, 50, nothing)
    # Create a trackbar for x position
    cv2.createTrackbar("x shift", "image", 100, 1500, nothing)
    # Create a trackbar for y position
    cv2.createTrackbar("y shift", "image", 50, 1000, nothing)

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
    length = 25

    orth_x = 10 / 5
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

    # Kernal for opening
    kernal = np.ones((5, 5), np.uint8)

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

        # Opening
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal)

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

def find_outliers(contours, data_area, sensitivity=3):
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
    outliers_areas = []
    # Remove the upper and lower bound outliers form the data and contours append outliers contour to outliers list
    for i in range(len(data_area)):
        if (data_area[i] > u_bound) or (data_area[i] < l_bound):
            # Creates outliers list
            outliers_contours.append(contours[i])
            # Creates outliers area list
            outliers_areas.append(data_area[i])

        else:
            # Creates inliers list
            inliers_contours.append(contours[i])
            # Creates inliers area list
            inliers_areas.append(data_area[i])


    # Calculate the number of inliers and outliers
    num_inliers = len(inliers_contours)
    num_outliers = len(outliers_contours)

    return tuple(inliers_contours), tuple(outliers_contours), inliers_areas, outliers_areas, num_inliers, num_outliers


def analysis(frame, threshold, factor, x_shift, y_shift, frame_no, pixel_area_ratio):
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

    # Kernal for opening
    kernal = np.ones((5, 5), np.uint8)

    # Opening
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Total number of found contours
    num_contours = len(contours)

    # Find pixel area of each contour
    raw_contour_area = []
    for j in contours:
        area = cv2.contourArea(j)
        raw_contour_area.append(area)

    if num_contours > 1:
        # Find the outliers
        contours, outliers_contours, inliers_areas, outliers_areas, num_inliers, num_outliers = find_outliers(contours, raw_contour_area)
    else:
        contours = contours
        outliers_contours = []
        inliers_areas = raw_contour_area
        outliers_areas = []
        num_inliers = num_contours
        num_outliers = 0

    # Draw contours inliers
    cv2.drawContours(image_trans, contours, -1, (0, 255, 0), 1)
    # Draw contours outliers
    cv2.drawContours(image_trans, outliers_contours, -1, (0, 0, 255), 1)

    # relates number of pixels to area in m^2
    # pixel_area_ratio = 1 / 700

    # Catagarise contours into pancake, brash and frazil
    pancake = 0
    pancake_area = 0
    brash = 0
    brash_area = 0
    frazil = 0
    frazil_area = 0

    for k in range(len(inliers_areas)):
        # Convert pixel area to m^2
        area = inliers_areas[k] * pixel_area_ratio

        if area > 2:
            pancake += 1
            pancake_area += area
        elif area > 1:
            brash += 1
            brash_area += area
        elif area > 1/7:
            frazil += 1
            frazil_area += area


    # Calculate percentage of outliers
    if num_contours == 0:
        percentage_outliers = 0
    else:
        percentage_outliers = round((num_outliers/num_contours)*100, 2)
    # Calculate sea ice concentration neglecting outliers
    SIC = round(100 * (sum(raw_contour_area) - sum(outliers_areas)) / (((image_trans.shape[0]) * image_trans.shape[1]) - sum(outliers_areas)), 2)
    # Extra data
    time = '2023-09-30 12:00:00'  # Replace with actual time from camera
    geo_location = ' 71.6734° N, 2.8451° W'  # Replace with actual location from camera

    cv2.putText(frame, str("Number of contours = " + str(num_inliers)), (width-450, 60), font, 0.8, (0, 0, 255))
    cv2.putText(frame, str("Percentage of outliers = " + str(percentage_outliers)), (width - 450, 90), font, 0.8, (0, 0, 255))
    cv2.putText(frame, str("Ice concentration in % =" + str(SIC)), (width-450, 120), font,0.8, (0, 0, 255))

    log_message = f"Frame {frame_no}: Num_Contours={num_contours}, Percentage_outliers={percentage_outliers}, Concentration={SIC}, Pancake_no={pancake}, Brash_no={brash}, Frazil_no={frazil}, Pancake_area={round(pancake_area,2)}, Brash_area={round(brash_area, 2)}, Frazil_area={round(frazil_area, 2 )}, Time={time}, Geo_location={geo_location}"
    logging.info(log_message)

    return frame, image_trans


def create_video(videofile, threshold, factor, x_shift, y_shift, trans_width,  trans_height, pixel_area_ratio):
    """
    This function is used to create the video with the analysis done on it
    :param videofile: the path of the video file
    :param threshold: the threshold value taken from the set_threshold function
    :param factor: the scaling factor taken from the initial_perspective_transform function
    :param x_shift: the x shift number of pixels taken from the initial_perspective_transform function
    :param y_shift: the y shift number of pixels taken from the initial_perspective_transform function
    :param trans_width: the width of the transformed image taken from the initial_perspective_transform function
    :param trans_height: the height of the transformed image taken from the initial_perspective_transform function
    :param pixel_area_ratio: the pixel area ratio taken from the calibration function
    :return: the video with the analysis done on it
    """
    n_frames, height, width, fps = get_meta_data(videofile)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out1 = cv2.VideoWriter("out_test05.mp4", fourcc, 10, (width + int(trans_width * height / trans_height), height), True)
    cap1 = cv2.VideoCapture(videofile)
    for i in tqdm(range(n_frames), total=n_frames):
        ret, img = cap1.read()
        if ret == False:
            break

        # Analysis
        img, image_trans = analysis(img, threshold, factor, x_shift, y_shift, (i+1), pixel_area_ratio)
        image_trans = cv2.resize(image_trans, (int(trans_width * height / trans_height), height))

        # Concatenate 'img' and 'image_trans' horizontally
        combined_frame = cv2.hconcat([img, image_trans])

        # Write the concatenated frame to the output video file
        out1.write(combined_frame)

    out1.release()
    cap1.release()

def plotting():
    # Read the data from the log file
    log_lines = []
    frames = []
    SIC = []
    pancake = []
    brash = []
    frazil = []
    pancake_area = []
    brash_area = []
    frazil_area = []

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
        SIC_start = line.find("Concentration=") + 14
        SIC_end = line.find(", Pancake_no")
        SIC_f = float(line[SIC_start:SIC_end])
        SIC.append(SIC_f)
        pancake_start = line.find("Pancake_no=") + 11
        pancake_end = line.find(", Brash_no")
        pancake_f = float(line[pancake_start:pancake_end])
        pancake.append(pancake_f)
        brash_start = line.find("Brash_no=") + 9
        brash_end = line.find(", Frazil_no")
        brash_f = float(line[brash_start:brash_end])
        brash.append(brash_f)
        frazil_start = line.find("Frazil_no=") + 10
        frazil_end = line.find(", Pancake_area")
        frazil_f = float(line[frazil_start:frazil_end])
        frazil.append(frazil_f)
        pan_area_start = line.find("Pancake_area=") + 13
        pan_area_end = line.find(", Brash_area")
        pan_area_f = float(line[pan_area_start:pan_area_end])
        pancake_area.append(pan_area_f)
        brash_area_start = line.find("Brash_area=") + 11
        brash_area_end = line.find(", Frazil_area")
        brash_area_f = float(line[brash_area_start:brash_area_end])
        brash_area.append(brash_area_f)
        frazil_area_start = line.find("Frazil_area=") + 12
        frazil_area_end = line.find(", Time")
        frazil_area_f = float(line[frazil_area_start:frazil_area_end])
        frazil_area.append(frazil_area_f)

    frames = np.array(frames)
    SIC = np.array(SIC)
    pancake = np.array(pancake)
    brash = np.array(brash)
    frazil = np.array(frazil)
    pancake_area = np.array(pancake_area)
    brash_area = np.array(brash_area)
    frazil_area = np.array(frazil_area)

    # Find the percentage distribution of ice floe types by number
    pancake_distribution = np.mean(pancake / (pancake + brash + frazil) * 100)
    brash_distribution = np.mean(brash / (pancake + brash + frazil) * 100)
    frazil_distribution = np.mean(frazil / (pancake + brash + frazil) * 100)

    # Plot bars in stack manner
    plt.bar(frames, pancake, color='r')
    plt.bar(frames, brash, bottom=pancake, color='b')
    plt.bar(frames, frazil, bottom=pancake + brash, color='y')
    plt.xlabel("Frames")
    plt.ylabel("Number of floes")
    plt.legend([f"Pancake ({pancake_distribution:.2f}%)", f"Brash ({brash_distribution:.2f}%)", f"Frazil ({frazil_distribution:.2f}%)"])
    plt.title("Distribution of ice floe types by number of floes")
    plt.show()

    # Find the percentage distribution of ice floe types by number
    pancake_area_distribution = np.mean(pancake_area / (pancake_area + brash_area + frazil_area) * 100)
    brash_area_distribution = np.mean(brash_area / (pancake_area + brash_area + frazil_area) * 100)
    frazil_area_distribution = np.mean(frazil_area / (pancake_area + brash_area + frazil_area) * 100)

    # Plot bars in stack manner
    plt.bar(frames, pancake_area, color='r')
    plt.bar(frames, brash_area, bottom=pancake_area, color='b')
    plt.bar(frames, frazil_area, bottom=pancake_area + brash_area, color='y')
    plt.xlabel("Frames")
    plt.ylabel("Total area of floes [m^2]")
    plt.legend([f"Pancake ({pancake_area_distribution:.2f}%)", f"Brash ({brash_area_distribution:.2f}%)", f"Frazil ({frazil_area_distribution:.2f}%)"])
    plt.title("Distribution of ice floe types by area [m^2] of floes")
    plt.show()

    SIC_ave = []
    frames_ave = []
    average = np.full((len(frames)), np.average(SIC))
    # Average sea ice concentation over 10 frames
    for i in range(math.floor(len(frames)/10)):
        SIC_ave.append(sum(SIC[10*i:10*(i+1)])/10)
        frames_ave.append(10*(i+1))

    # Plot the data using matplotlib
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(frames, SIC, label='Sea Ice Concentration per frame', color='b')
    plt.plot(np.array(frames_ave), np.array(SIC_ave), label='Sea Ice Concentration averaged over 10 frames', color='r')
    plt.plot(frames, average, label=f'Average Sea Ice Concentration ({np.average(SIC):.2f}%)', color='g')
    plt.yticks(np.arange(0, 101, 10))
    plt.xlabel('Frame number')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title('Sea Ice Data Over Time')
    plt.show()


def capture_image():
    """
    This function captures an image from the camera and saves it as 'captured_image.jpg'
    Press y to capture the image and n to exit the function
    """
    frame_path = "captured_image.jpg"
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Prompt the user for input ('y' for capturing, 'n' for exiting)
        key = cv2.waitKey(1)

        if key == ord('y'):
            # Save the captured image
            cv2.imwrite("captured_image.jpg", frame)
            print("Image captured and saved as 'captured_image.jpg'")
        elif key == ord('n'):
            print("Image capture cancelled")
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return frame, frame_path