import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


def get_meta_data(videofile):
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
    video_cap = cv2.VideoCapture(videofile)
    success, image = video_cap.read()
    if success:
        cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG file

    return image

def nothing(x):
    pass

def initial_perspective_transform(imagefile, image):
    # Sizing in original image
    h, w, c = image.shape

    x = int(w/2)
    y = int(h/5)
    length = 125

    orth_x = 14/5
    orth_y = 214/125

    # Create a black image, a window
    # Output image size
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

    # Create a trackbar for scale
    cv2.createTrackbar("scale", "image", 1, 5, nothing)
    # Create a trackbar for x position
    cv2.createTrackbar("x shift", "image", 100, 700, nothing)
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

    return image_trans, factor, x, y

def perspective_transform(image, factor, x, y):
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

    # Draw lines to combine the points
    cv2.line(image, (int(T_L[0]), int(T_L[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)
    cv2.line(image, (int(B_L[0]), int(B_L[1])), (int(T_L[0]), int(T_L[1])), (0, 0, 255), 2)
    cv2.line(image, (int(B_R[0]), int(B_R[1])), (int(B_L[0]), int(B_L[1])), (0, 0, 255), 2)
    cv2.line(image, (int(B_R[0]), int(B_R[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)

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

    return image_trans

def set_threshold():

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

def analysis_video(videofile, threshold):
    i = 1  # Images file counter
    cap = cv2.VideoCapture(videofile)

    # Keep iterating break
    while True:
        ret, frame = cap.read()  # Read frame from first video

        if ret:
            # Perspective transform
            image_trans = perspective_transform(frame)

            # Select the font
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Convert to grayscale
            imgray = cv2.cvtColor(image_trans, cv2.COLOR_BGR2GRAY)

            # Find threshold
            ret, thresh = cv2.threshold(imgray, threshold, 4, 0)  # detect only the specified range

            # Find contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            con_area = []
            for j in contours:
                area = cv2.contourArea(j)

                if area > 200:
                    con_area.append(area / 700)

            cv2.putText(frame, str("Number of contours = " + str(len(con_area))), (800, 100), font, 0.8, (0, 0, 255))
            cv2.putText(frame, str("Ice concentration in % =" + str(round(100 * (sum(con_area)) / ((image_trans.shape[0]) * image_trans.shape[1] / 700), 2))), (800, 130), font, 0.8,(0, 0, 255))
            cv2.imwrite(str(i) + '.jpg', frame)  # Write frame to JPEG file (1.jpg, 2.jpg, ...)

            i += 1  # Advance file counter
        else:
            # Break the internal loop when res status is False.
            break

    cap.release()  # Release must be inside the outer loop
    cv2.destroyAllWindows()

def analysis(frame, threshold, factor, x_shift, y_shift):
    # Perspective transform
    image_trans = perspective_transform(frame, factor, x_shift, y_shift)

    # Select the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Convert to grayscale
    imgray = cv2.cvtColor(image_trans, cv2.COLOR_BGR2GRAY)

    # Find threshold
    ret, thresh = cv2.threshold(imgray, threshold, 4, 0)  # detect only the specified range

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    con_area = []
    for j in contours:
        area = cv2.contourArea(j)

        if area > 200:
            con_area.append(area / 700)

    cv2.putText(frame, str("Number of contours = " + str(len(con_area))), (400 + x_shift, 100), font, 0.8, (0, 0, 255))
    cv2.putText(frame, str("Ice concentration in % =" + str(
        round(100 * (sum(con_area)) / ((image_trans.shape[0]) * image_trans.shape[1] / 700), 2))), (400 + x_shift, 130), font,
                0.8, (0, 0, 255))

    return frame


def create_video(videofile, threshold, factor, x_shift, y_shift):
    n_frames, height, width, fps = get_meta_data(videofile)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("out_test03.mp4", fourcc, 10, (width, height), True)
    cap = cv2.VideoCapture(videofile)
    for frame in tqdm(range(n_frames), total=n_frames):
        ret, img = cap.read()
        if ret == False:
            break
        img = analysis(img, threshold, factor, x_shift, y_shift)
        out.write(img)
    out.release()
    cap.release()

def main():
    print("Start")

    # Get meta data
    n_frames, height, width, fps = get_meta_data('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video03.mp4')

    # Get first frame
    image = get_first_frame('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video03.mp4')

    # Perspective transform
    image_trans, factor, x_shift, y_shift = initial_perspective_transform("first_frame.jpg", image)

    # Set threshold
    threshold = set_threshold()

    # Analysis video
    create_video('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video03.mp4', threshold, factor, x_shift, y_shift)



if __name__ == "__main__":
    main()
