import gym
import tensorflow as tf
import numpy as py

class policy_gradient():
    def __init__(self):
        print("run init function.")



print(tf.__version__)
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
pg = policy_gradient()