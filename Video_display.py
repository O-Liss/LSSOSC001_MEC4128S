import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import IPython.display as ipd
from tqdm.notebook import tqdm
import subprocess

print("Start")

ipd.Video('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video01.mp4', width=500)
cap = cv2.VideoCapture('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/Visual /video01.mp4')

# Total number of frames in the video
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video height and width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# frame rate (frames per second)
fps = cap.get(cv2.CAP_PROP_FPS)

# Release the video capture object
#cap.release()

# returns the image and
ret, img = cap.read()
print("Returned", ret, "and the img shape is", img.shape)

# display opencv images in matplotlib
def display_cv2_img(img, figsize=(10,10)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis('off')

display_cv2_img(img)

fig, axs = plt.subplots(5, 5, figsize=(30,20))
axs = axs.flatten()


img_idx = 0
for frame in range(n_frames):
    ret, img = cap.read()
    if ret == False:
        break
    if frame % 25 == 0:
        axs[img_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[img_idx].set_title(f"Frame {frame}")
        axs[img_idx].axis('off')
        img_idx +=1

plt.tight_layout()

for frame in tqdm(range(n_frames)):
    ret, img = cap.read()
    if ret == False:
        break
    out.write(img)






