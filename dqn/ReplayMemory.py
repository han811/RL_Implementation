import os
import time
import random

import numpy as np

from Config import config

class ReplayMemory:
    """Replay Memory buffer"""
    def __init__(self, config):
        self.MemorySize = config['memory_size'] # memory size
        self.Observations = np.empty((self.MemorySize, config['height'], config['width']), dtype=np.uint8) # observation images
        self.Actions = np.empty(self.MemorySize, dtype=np.uint8) # action
        self.Rewards = np.empty(self.MemorySize, dtype=np.float32) # reward
        self.Terminals =  np.empty(self.MemorySize, dtype=np.bool) # check if terminal : 1 otherwise : 0
        
        self.ObservationDims = (config['height'], config['width'])
        
        self.Current = 0 # current index to write
        self.Count = 0 # total index to check the number of memory written
        
    def add(self, observation, action, reward, terminal):
        if observation.shape != self.ObservationDims:
            print(f"[ERROR]  Observation image size is not correct!!")
            return
        self.Observations[self.Current] = observation
        self.Actions[self.Current] = action
        self.Rewards[self.Current] = np.sign(reward) # positive reward +1, negative reward -1, otherwise 0
        self.Terminals[self.Current] = terminal
        
        self.Current = (self.Current + 1) % self.MemorySize
        if not self.Count == self.MemorySize:
            self.Count = self.Count + 1
        
    def getState(self, n):
        
        c = n % self.MemorySize
        
        states = [self.Observations[c-4:c, ...]]

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))


        return states, self.Actions[c-4:c], self.Rewards[c-4:c], self.Terminals[c-4:c]
            
    def getStateMiniBatch(self, batch_size=32):
        if self.Count < 4:
            print(f"[ERROR]  not enough memory for mini_batch!!")
            return
        else:
            idxs = []
            for i in range(batch_size):
                while True:
                    idx = random.randint(4, self.Count - 1)
                    if idx >= self.Current and idx - 4 <= self.Current:
                        continue
                    if self.Terminals[idx - 4:idx].any():
                        continue
                    break
                idxs.append(idx)

            states = []
            new_states = []
            for i in idxs:
                states.append(self.Observations[i-4:i, ...])
                new_states.append(self.Observations[i-3:i+1, ...])

            states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
            new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

            return states, self.Actions[idxs], self.Rewards[idxs], new_states, self.Terminals[idxs]
            
    
if __name__ == "__main__":
    a = np.ones((84,84,4))
    m = ReplayMemory(config)
    m.add(a,a,0,0,0)
    s , _ , _ , _ = m.getState(0)
    print(s)           
        