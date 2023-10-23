# Description: This script is used to capture the video data from the camera and save it as a .mp4 file.
from SEA_ICE_Functions import *


def main():
    print("Start")
    # Calibration pixel area ratio use Calibration.py
    pixel_area_ratio = 1/700

    path = "/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Thermal/output_video_thermal_aqi2_0_cut.mp4"

    # Get meta data
    n_frames, height, width, fps = get_meta_data(path)
    print("The number of frames is:", n_frames)
    print("The height of the video is:", height)
    print("The width of the video is:", width)

    # Get first frame
    image = get_first_frame(path)

    # Set perspective transform
    image_trans, factor, x_shift, y_shift, trans_width,  trans_height = initial_perspective_transform("first_frame.jpg")

    # Set threshold
    threshold = set_threshold()

    # Log the start time
    start_time = time.time()

    # Analysis video
    create_video(path, threshold, factor, x_shift, y_shift, trans_width, trans_height, pixel_area_ratio)

    # Log the end time
    end_time = time.time()

    print("The time taken to analyse the video is:", end_time - start_time)
    print("The average time per frame is:", (end_time - start_time)/n_frames)

    plotting()


if __name__ == "__main__":
    main()
