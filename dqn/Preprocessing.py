import os
import random
import time

import numpy as np
import cv2
from matplotlib import pyplot as plt

from Utils import rgb2gray

def preprocess(image, dims=(84,110), debug=False):
    image = rgb2gray(image)
    image = image.astype(np.uint8)
    image = cv2.resize(image, dsize=dims, interpolation=cv2.INTER_CUBIC)
    image = image[20:20+dims[0],...]
    if debug:
        # plt.imshow(np.reshape(image,(84,84)))
        plt.imshow(image)
        plt.show()
    
    return image

if __name__ == "__main__":
    a = np.empty((210,160,3))
    a = preprocess(a,debug=False)
    print(a.shape)