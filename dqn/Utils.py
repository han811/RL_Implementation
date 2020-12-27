import os
import sys
import time

import numpy as np

# numpy utils
def save_npy(obj, path):
    np.save(path, obj)
    print(f"[NOTICE]  save {path}")
    
def load_npy(path):
    obj = np.load(path)
    print(f"[NOTICE]  load {path}")
    return obj    

# For DQN
# reference : https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems
def rgb2gray(image):
  return np.dot(image[...,:3], [0.299, 0.587, 0.114])

