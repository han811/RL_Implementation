import os
import random
import time

import numpy as np
import cv2

from Utils import rgb2gray

def preprocess(image, dims):
    image = rgb2gray(image) / 255.0
    image = cv2.resize(image, dsize=dims, interpolation=cv2.INTER_CUBIC)
    image = np.reshape(image, (dims[0],dims[1],1))
    if dims[0]>=dims[1]:
        d = (int)((dims[0] - dims[1]) / 2)
        image = image[d:d+dims[1],...]
    else:
        d = (int)((dims[1] - dims[0]) / 2)
        image = image[:,d:d+dims[0],...]
    return image

if __name__ == "__main__":
    a = np.empty((210,160,3))
    a = preprocess(a,(110,84))
    print(a.shape)