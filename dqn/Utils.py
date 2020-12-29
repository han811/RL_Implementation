import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

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


def gpuCheck():
    if tf.config.list_physical_devices('GPU'):
        print("Your gpu is ready!!")
    else:
        raise ValueError("Your gpu is not ready for learning!!")

if __name__ == "__main__":
    gpuCheck()
