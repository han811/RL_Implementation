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
    def __init__(self, config):
        self.Model = DQN()
        self.Preprocess = preprocess
        self.ReplayMemory = ReplayMemory(config)   
        self.Env = gym.make(config['env_name'],frameskip=4)
        self.Env.render()
        self.eps = 1.0
        self.Current = 0
        self.Optim = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        self.discount_factor = config['discount_factor']

    def reset(self):
        self.Env.reset()
        self.Env.render()

    def policy(self, inputs):
        inputs = inputs / 255.
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
        
        obs, reward, done, info = self.Env.step(action)
        self.Env.render()
        obs = preprocess(obs)
        self.ReplayMemory.add(obs, action, reward, done)
        self.Current += 1
        self.Current = self.Current % self.ReplayMemory.MemorySize
            

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
                self.Env.render()
                obs = preprocess(obs)
                self.ReplayMemory.add(obs, action, reward, done)
                self.Current += 1
                self.Current = self.Current % self.ReplayMemory.MemorySize
                if done:
                    sig = False
            # print(self.tmp.shape)
            # exit()
            if sig == False:
                self.reset()
                continue
            
            c_l = 0
            while True:
                s , _ , _ , _ = self.ReplayMemory.getState(self.Current)
                # print(s.shape)
                # exit()
                obs, reward, done, info = self.policy(s)
                if done:
                    break
                
                s1 , a , r , s2 , _ = self.ReplayMemory.getStateMiniBatch(32)
                y = []
                A = []
                for i in range(32):
                    aidx = [False,False,False,False]
                    aidx[a[i]] = True
                    A.append(aidx.copy())
                # t1 = time.time()
                
                actions = self.Model.predict(s2/255.)
                # t2 = time.time()
                # print(t2-t1)
                for i in range(32):
                    action = actions[i].argmax()
                    # print(action)
                    # print(actions[i])
                    y.append(r[i] + self.discount_factor * actions[i][action])
                # print(len(inputs))
                # print(inputs[0].shape)
                # exit()
                loss = self.update(s1/255., y, A, mini_batch)
                if c_l % 100 == 0:
                    print(f"loss: {loss}")
                c_l+=1

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
    agent.train()