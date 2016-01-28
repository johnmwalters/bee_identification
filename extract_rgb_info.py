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

def extract_rgb_info(rgb, ax=None):
    """Extract color statistics as features:
        - pixel values (flattened)
        - X, Y sums per channel
        - percentiles per channel
        - percentile diffs per channel

        Plots if ax is passed
    """
    # toss alpha if it exists
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    # start with raw pixel values as features
    features = [rgb.flatten()]

    # add some basic statistics on the color channels (R, G, B)
    for channel in range(3):
        this_channel = rgb[:, :, channel].astype(np.float)
        sums = np.hstack([this_channel.sum(),
                          this_channel.sum(axis=0),
                          this_channel.sum(axis=1)])

        # percentiles
        ps = [1, 3, 5, 10, 50, 90, 95, 97, 99]
        percentiles = np.array(np.percentile(this_channel, ps))
        diff = percentiles[-4:] - percentiles[:4]
        
        # plot if we have been passed an axis
        if ax is not None:
            channel_name = ['r', 'g', 'b'][channel]       
            sns.kdeplot(this_channel.flatten(),
                        ax=ax,
                        label=channel_name,
                        color=channel_name)
            ax.set_title("Color channels")

        
        # store the features for this channel
        features += [sums, percentiles, diff]
        
    # return all the color features as a flat array
    return np.hstack(features).flatten()