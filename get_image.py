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