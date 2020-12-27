import os
import time

import numpy as np
import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self._build()
        
    def _build(self):
        self.ConvLayer1 = tf.keras.layers.Conv2D(32, (8,8), (4,4), activation=tf.keras.activations.relu, name='conv_layer_1')
        self.ConvLayer2 = tf.keras.layers.Conv2D(64, (4,4), (2,2), activation=tf.keras.activations.relu, name='conv_layer_2')
        self.ConvLayer3 = tf.keras.layers.Conv2D(64, (3,3), (1,1), activation=tf.keras.activations.relu, name='conv_layer_3')
        self.Flatten = tf.keras.layers.Flatten()
        self.FCLayer1 = tf.keras.layers.Dense(512, activation=tf.keras.activations.relu, name='fc_layer_1')
        self.FCLayer2 = tf.keras.layers.Dense(4, name='fc_layer_2')
    
    @tf.function 
    def call(self, inputs):
        x = self.ConvLayer1(inputs)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.Flatten(x)
        x = self.FCLayer1(x)
        x = self.FCLayer2(x)
        return x
    
    def test(self, input_shape=(84,84,4)):
        x1 = tf.keras.layers.Input(shape=input_shape)
        x = self.ConvLayer1(x1)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.Flatten(x)
        x = self.FCLayer1(x)
        x = self.FCLayer2(x)
        return tf.keras.Model(inputs=[x1], outputs=[x])
    
if __name__ == "__main__":
    model = DQN()
    model.test((84,84,4)).summary()
        
        
        
        
        
        
        
        