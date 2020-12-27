import os
import time
import random

import numpy as np
import tensorflow as tf
import gym

from ReplayMemory import ReplayMemory
from Model import DQN
from Preprocessing import preprocess
from Config import config

class DQN_train:
    def __init__(self, config, name='Breakout-v0'):
        self.Model = DQN()
        self.Preprocess = preprocess
        self.ReplayMemory = ReplayMemory(config)   
        self.Env = gym.make(name)
        self.Env.render()
        self.Current = None
        self.Steps = 0
        self.eps = 1.0
        self.Optim = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        self.tmp = np.empty((84,84,4), dtype=np.float32)
        
    def reset(self):
        self.Current = None
        self.Steps = 0
        self.ReplayMemory.reset()
        self.Env.reset()
        self.Env.render()

    def policy(self, inputs, k=4):
        actions = self.Model.predict(inputs)
        action = actions.argmax()
        # print(actions)
        # print(action)
        # exit()
        e = random.uniform(0,1)
        if not e < 1-self.eps:
            a = [0,1,2,3]
            a.remove(action)
            action = random.choice(a)
        
        for _ in range(k):
            obs, reward, done, info = self.Env.step(action)
            if reward>0:
                reward = 1
            elif reward<0:
                reward = -1
            else:
                reward = 0
            self.Env.render()
            self.Steps += 1
            self.Steps = self.Steps % self.ReplayMemory.MemorySize    
            obs = preprocess(obs,(84,84))
            self.tmp[...,0:3] = self.tmp[...,1:4]
            self.tmp[...,3:4] = obs.copy()
            self.Current = self.tmp.copy()
            self.ReplayMemory.add(self.Current, obs, action, reward, done)
        return obs, reward, done, info
        
    def train(self, mini_batch=32, k=4, episode=1000000):
        while(--episode>0):
            if episode<1e6:
                self.eps = (1-0.1) / 1e6
            else:
                self.eps = 0.1
            
            self.reset()
            
            sig = True
            
            for i in range(4):
                action = self.Env.action_space.sample()
                obs, reward, done, info = self.Env.step(action)
                if reward>0:
                    reward = 1
                elif reward<0:
                    reward = -1
                else:
                    reward = 0
                self.Env.render()
                obs = preprocess(obs,(84,84))
                self.tmp[...,i:i+1] = obs.copy()
                if done:
                    sig = False
            # print(self.tmp.shape)
            # exit()
            if sig == False:
                self.reset()
                continue
            self.Current = self.tmp.copy()
            self.ReplayMemory.add(self.Current, obs, action, reward, done)
            
            while(True):
                s , _ , _ , _ = self.ReplayMemory.getState(self.Steps)
                # print(s.shape)
                # exit()
                obs, reward, done, info = self.policy(s,k)
                if done:
                    break
                
                inputs = []
                y = []
                A = []
                
                for i in range(mini_batch):
                    idx = random.randint(0,self.Steps)
                    s1 , a , _ , _ = self.ReplayMemory.getState(idx-1)
                    s2 , _ , r , _ = self.ReplayMemory.getState(idx)
                    if i == 0:
                        inputs = s1.copy()
                    else:
                        inputs = np.concatenate((inputs, s1.copy()), axis=0)
                    aidx = [False,False,False,False]
                    aidx[a] = True
                    A.append(aidx.copy())
                    
                    actions = self.Model.predict(s2)
                    action = tf.argmax(actions, 0).numpy()[0]
                    # print(actions)
                    y.append(r + actions[0][action])
                # print(len(inputs))
                # print(inputs[0].shape)
                # exit()
                loss = self.update(inputs, y, A, mini_batch)
                print(f"loss: {loss}")

    # @tf.function
    def update(self, inputs, y, A, mini_batch=32):
        # print(inputs.dtype)
        # exit()
        with tf.GradientTape() as tape:
            out = self.Model(inputs)
            out = out * A
            out = tf.reduce_sum(out,1)
            loss = tf.keras.losses.MSE(y,out)
        loss_g = tape.gradient(loss, self.Model.trainable_variables)
        self.Optim.apply_gradients(zip(loss_g, self.Model.trainable_variables))
        return loss
            
            
            
                
                    
                
if __name__ == "__main__":
    # e = gym.make('Breakout-v0')
    # e.reset()
    # print(e.action_space.n)
    # print(preprocess(e.observation_space.high,(84,84)).shape)
    agent = DQN_train(config)
    # print(agent.Env.reward_range)
    agent.train()