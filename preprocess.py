import os
from tqdm import tqdm # smart progress bar
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc_params
import pandas as pd
import numpy as np
from skimage.feature import daisy
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist

from get_image import get_image
from extract_rgb_info import extract_rgb_info

def preprocess(img, demo=False):
    """ Turn raw pixel values into features.
    """
    
    def _demo_plot(img, stage="", is_ints=False, axes_idx=0):
        """ Utility to visualize the features we're building
        """
        if demo:
            axes[axes_idx].imshow(img / 255. if is_ints else img,
                                  cmap=bees_cm)
            axes[axes_idx].set_title(stage)
        return axes_idx + 1

    if demo:
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        axes = axes.flatten()
    
    # track which subplot we're plotting to
    axes_idx = 0
    axes_idx = _demo_plot(img, stage="Raw Image", is_ints=True, axes_idx=axes_idx)
        
    # FEATURE 1: Raw image and color data    
    if demo:
        color_info = extract_rgb_info(img, ax=axes[axes_idx])
        axes_idx += 1
    else:
        color_info = extract_rgb_info(img)
    
    # remove color information (hog and daisy only work on grayscale)
    gray = rgb2gray(img)
    axes_idx = _demo_plot(gray, stage="Convert to grayscale", axes_idx=axes_idx)
    
    # equalize the image
    gray = equalize_hist(gray)
    axes_idx = _demo_plot(gray, stage="Equalized histogram", axes_idx=axes_idx)
    
    # FEATURE 2: histogram of oriented gradients features
    hog_features = hog(gray,
                       orientations=12,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(1, 1),
                       visualise=demo)
    
    # if demo, we actually got a tuple back; unpack it and plot
    if demo:
        hog_features, hog_image = hog_features
        axes_idx = _demo_plot(hog_image, stage="HOG features", axes_idx=axes_idx)
        
    # FEATURE 3: DAISY features - sparser for demo so can be visualized
    params = {'step': 25, 'radius': 25, 'rings': 3} if demo \
             else {'step': 10, 'radius': 15, 'rings': 4}
    daisy_features = daisy(gray,
                           histograms=4,
                           orientations=8,
                           normalization='l1',
                           visualize=demo,
                           **params)
        
    if demo:
        daisy_features, daisy_image = daisy_features
        axes_idx = _demo_plot(daisy_image, stage="DAISY features", axes_idx=axes_idx)
    
    # return a flat array of the raw, hog and daisy features
    return np.hstack([color_info, hog_features, daisy_features.flatten()])