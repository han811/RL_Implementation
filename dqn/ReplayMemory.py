import os
import time
import random

import numpy as np

from Config import config

class ReplayMemory:
    def __init__(self, config):
        self.MemorySize = config['memory_size'] # memory size
        self.Observations = np.empty((self.MemorySize, config['height'], config['width'], 4), dtype=np.float32) # observation images
        self.NextObservations = np.empty((self.MemorySize, config['height'], config['width'], 4), dtype=np.float32) # next observation images
        self.Actions = np.empty(self.MemorySize, dtype=np.uint8) # action
        self.Rewards = np.empty(self.MemorySize, dtype=np.int8) # reward
        self.Terminals =  np.empty(self.MemorySize, dtype=np.bool) # check if terminal : 1 otherwise : 0
        
        self.ObservationDims = (config['height'], config['width'], 4)
        
        self.Current = 0
        self.Count = 0
        
    def add(self, observation, next_observation, action, reward, terminal):
        # print(observation.shape)
        # print(self.ObservationDims)
        if observation.shape != self.ObservationDims:
            print(f"[ERROR]  Observation image size is not correct!!")
            return
        self.Observations[self.Current] = observation
        self.NextObservations[self.Current] = next_observation
        self.Actions[self.Current] = action
        self.Rewards[self.Current] = reward
        self.Terminals[self.Current] = terminal
        
        self.Current = (self.Current + 1) % self.MemorySize
        if not self.Count == self.MemorySize:
            self.Count = self.Count + 1
            
    def reset(self):
        self.Observations = np.empty((self.MemorySize, self.ObservationDims[0], self.ObservationDims[1], 4), dtype=np.float32) # observation images
        self.NextObservations = np.empty((self.MemorySize, self.ObservationDims[0], self.ObservationDims[1], 4), dtype=np.float32) # next observation images
        self.Actions = np.empty(self.MemorySize, dtype=np.uint8) # action
        self.Rewards = np.empty(self.MemorySize, dtype=np.integer) # reward
        self.Terminals =  np.empty(self.MemorySize, dtype=np.bool) # check if terminal : 1 otherwise : 0        
        self.Current = 0
        self.Count = 0
        
    def getState(self, n):
        
        c = n % self.MemorySize
        
        sample_observations = self.Observations[c:c+1].copy()
        sample_actions = self.Actions[c].copy()
        sample_rewards = self.Rewards[c].copy()
        sample_next_observations = self.NextObservations[c:c+1].copy()
        
        return sample_observations, sample_actions, sample_rewards, sample_next_observations
    
if __name__ == "__main__":
    a = np.ones((84,84,4))
    m = ReplayMemory(config)
    m.add(a,a,0,0,0)
    s , _ , _ , _ = m.getState(0)
    print(s)           
        