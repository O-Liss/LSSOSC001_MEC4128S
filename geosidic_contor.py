import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.color import rgb2gray
from skimage.util import img_as_float



def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


# Load the grayscale image
im = cv2.imread('/Users/oscarliss/Desktop/LSSOSC001_MEC4128S/Practice data/IMG_1485.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocess the image to highlight contours
gimage = inverse_gaussian_gradient(im)

# Initial starting pionts for the snake
init_ls = np.zeros(im.shape, dtype=np.int8)
init_ls[600:-500, 600:-500] = 1

# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)

# Morphological GAC, balloon
ls = morphological_geodesic_active_contour(gimage, 200, init_ls, smoothing=1, balloon=5, threshold=0.7, iter_callback=callback)

fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(im, cmap="gray")
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors='b')
ax[0].set_title("Morphological GAC segmentation", fontsize=12)

ax[1].imshow(ls, cmap="gray")
ax[1].set_axis_off()
contour = ax[1].contour(evolution[0], [0.5], colors='r')
contour.collections[0].set_label("Starting Contour")
contour = ax[1].contour(evolution[5], [0.5], colors='g')
contour.collections[0].set_label("Iteration 5")
contour = ax[1].contour(evolution[-1], [0.5], colors='b')
contour.collections[0].set_label("Last Iteration")
ax[1].legend(loc="upper right")
title = "Morphological GAC Curve evolution"
ax[1].set_title(title, fontsize=12)

plt.show()