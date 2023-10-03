import numpy as np
import cv2

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
        cv2.imwrite("../first_frame.jpg", image)  # save frame as JPEG file

    return image

def nothing(x):
    pass

def initial_perspective_transform(imagefile, image):
    # Sizing in original image
    h, w, c = image.shape

    x = int(w/2)
    y = int(h/5)
    factor = 2
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

    return image_trans, factor, height, width



def main():
    print("Start")

    # Get meta data
    n_frames, height, width, fps = get_meta_data('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video03.mp4')

    # Get first frame
    image = get_first_frame('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video03.mp4')

    # Perspective transform
    image_trans, factor, height, width = initial_perspective_transform("first_frame.jpg", image)



if __name__ == "__main__":
    main()