import os
import time
import random

import numpy as np
import tensorflow as tf
from collections import deque
import gym

from Model import DQN
from Preprocessing import preprocess
from Config import config

class DQN_train:
    def __init__(self, config):
        # parameter setting
        self.StateSize = config['state_size']
        self.ActionSize = config['action_size']
        self.DiscountFactor = config['discount_factor']
        self.LearningRate = config['learning_rate']

        self.Eps = 1.0
        self.EpsStart = 1.0
        self.EpsEnd = 0.1
        self.EpsSteps = 1000000
        self.EpsDecayRate = (self.EpsStart - self.EpsEnd) / self.EpsSteps
        self.BatchSize = config['mini_batch_size']
        self.StartLearningStep = config['start_learning_step']
        self.TargetModelUpdateFreq = config['target_model_update_freq']

        self.ReplayMem = deque(maxlen=config['memory_size'])
        self.NoOpSteps = 30

        self.Model = DQN(config['state_size'], config['action_size'])
        self.TargetModel = DQN(config['state_size'], config['action_size'])
        if config['reward_clip']:
            self.Optim = tf.keras.optimizers.Adam(learning_rate=self.LearningRate, clipnorm=10.)
        else:
            self.Optim = tf.keras.optimizers.Adam(learning_rate=self.LearningRate)

        self.updateModel()

        self.AvgQMax , self.AvgLoss = 0, 0

        self.Writer = tf.summary.create_file_writer('summary/breakout_dqn')
        self.ModelPath = os.path.join(os.getcwd(), 'save_model', 'model')

    def updateModel(self):
        self.TargetModel.set_weights(self.Model.get_weights())

    def getAction(self, inputs):
        inputs = inputs / 255.
        if np.random.rand() <= self.Eps:
            return random.randrange(self.ActionSize)
        else:
            actions = self.Model(inputs)
            return np.argmax(actions[0])

    def addMemory(self, s1, action, reward, s2, done):
        self.ReplayMem.append((s1, action, reward, s2, done))

    def drawTensorBoard(self, score, step, episode):
        with self.Writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Max Q/Episode', self.AvgQMax/float(step), step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
            tf.summary.scalar('Average Loss/Episode', self.AvgLoss/float(step), step=episode)
            
    def trainModel(self):
        if self.Eps > self.EpsEnd:
            self.Eps -= self.EpsDecayRate
        
        sample = random.sample(self.ReplayMem, self.BatchSize)

        s1 = np.array([s[0][0] / 255. for s in sample])
        a = np.array([s[1] for s in sample])
        r = np.array([s[2] for s in sample])
        s2 = np.array([s[3][0] / 255. for s in sample])
        done = np.array([s[4] for s in sample])

        with tf.GradientTape() as tape:
            Q = self.Model(s1)
            mask = tf.one_hot(a, self.ActionSize)
            Q = tf.reduce_sum(mask * Q, axis=1)

            Q2 = self.TargetModel(s2)

            maxQ2 = np.amax(Q2, axis=1)
            y = r + (1-done) * self.DiscountFactor * maxQ2

            loss = tf.keras.losses.MSE(y, Q)
            self.AvgLoss += loss.numpy()

        grad = tape.gradient(loss, self.Model.trainable_variables)
        self.Optim.apply_gradients(zip(grad, self.Model.trainable_variables))

    def train(self):
        
        env = gym.make(config['env_name'])
        
        total_steps = 0
        score_avg = 0
        score_max = 0

        num_episode = 50000

        for e in range(num_episode):
            done = False
            dead = False

            step, score, start_life = 0, 0, 5

            obs = env.reset()

            for _ in range(random.randint(1, self.NoOpSteps)):
                obs, _ , _ , _ = env.step(1)

            s = preprocess(obs)
            h = np.stack((s ,s, s, s), axis=2)
            h = np.reshape([h], (1,84,84,4))

            while not done:
                # env.render()
                total_steps += 1
                step += 1

                action = self.getAction(h)
                action_idx = action+1

                if dead:
                    action, action_idx, dead = 0, 1, False
                
                obs, reward, done, info = env.step(action_idx)

                n_s = preprocess(obs)
                n_s = np.reshape([n_s], (1,84,84,1))
                n_h = np.append(n_s, h[:,:,:,:3], axis=3)

                self.AvgQMax += np.amax(self.Model(h/255.)[0])

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                
                score += reward
                reward = np.sign(reward)

                self.addMemory(h, action, reward, n_h, dead)



                if len(self.ReplayMem) >= self.StartLearningStep:
                    self.trainModel()
                    if total_steps % self.TargetModelUpdateFreq == 0:
                        self.updateModel()

                    if dead:
                        h = np.stack((n_s,n_s,n_s,n_s), axis=2)
                        h = np.reshape([h], (1,84,84,4))
                    else:
                        h = n_h
                    
                    if done:
                        if total_steps > self.StartLearningStep:
                            self.drawTensorBoard(score, step, e)
                        
                        score_avg = 0.9*score_avg + 0.1*score if score_avg!=0 else score
                        score_max = score if score > score_max else score_max

                        log = "episode: {:5d} | ".format(e)
                        log += "score: {:4.1f} | ".format(score)
                        log += "score max: {:4.1f} | ".format(score_max)
                        log += "score avg: {:4.1f} | ".format(score_avg)
                        log += "memory size: {:5d} | ".format(len(self.ReplayMem))
                        log += "epsilon: {:.3f} | ".format(self.Eps)
                        log += "q_avg: {:3.2f} | ".format(self.AvgQMax/step)
                        log += "avg_loss: {:3.2f} | ".format(self.AvgLoss/step)
                        print(log)

                        self.AvgQMax, self.AvgLoss = 0, 0

                    if e % 1000 == 0:
                        self.Model.save_weights("./save_model/model", save_format="tf")





                
if __name__ == "__main__":
    agent = DQN_train(config)
    agent.train()
