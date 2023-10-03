# Description: This script is used to capture the video data from the camera and save it as a .mp4 file.
from SEA_ICE_Functions import *

def main():
    print("Start")

    # Get meta data
    n_frames, height, width, fps = get_meta_data('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video01.mp4')

    # Get first frame
    image = get_first_frame('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video01.mp4')

    # Perspective transform
    image_trans, factor, x_shift, y_shift, trans_width,  trans_height = initial_perspective_transform("first_frame.jpg")

    # Set threshold
    threshold = set_threshold()

    # Analysis video
    create_video('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video01.mp4', threshold, factor, x_shift, y_shift, trans_width, trans_height)




if __name__ == "__main__":
    main()
