import os
from tqdm import tqdm
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rc_params
from skimage.feature import daisy
from skimage.feature import hog
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc


def get_image(row_or_str, root="data/images/"):
    # if we have an instance from the data frame, pull out the id
    # otherwise, it is a string for the image id
    if isinstance(row_or_str, pd.core.series.Series):
        row_id = row_or_str.name
    else:
        row_id = row_or_str
    
    filename = "{}.jpg".format(row_id)
    
    # test both of these so we don't have to specify. Image should be
    # in one of these two. If not, we let Image.open raise an exception.
    train_path = os.path.join(root, "train", filename)
    test_path = os.path.join(root, "test", filename)
    
    file_path = train_path if os.path.exists(train_path) else test_path
    
    return np.array(Image.open(file_path), dtype=np.int32)

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
    
def create_feature_matrix(label_dataframe):
    n_imgs = label_dataframe.shape[0]
    
    # initialized after first call to 
    feature_matrix = None
    
    for i, img_id in tqdm(enumerate(label_dataframe.index)):
        features = preprocess(get_image(img_id))
        
        # initialize the results matrix if we need to
        # this is so n_features can change as preprocess changes
        if feature_matrix is None:
            n_features = features.shape[0]
            feature_matrix = np.zeros((n_imgs, n_features), dtype=np.float32)
            
        if not features.shape[0] == n_features:
            print "Error on image {}".format(img_id)
            features = features[:n_features]
        
        feature_matrix[i, :] = features
        
    return feature_matrix