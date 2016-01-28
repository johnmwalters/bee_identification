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
#from extract_rgb_info import extract_rgb_info
from preprocess import preprocess

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