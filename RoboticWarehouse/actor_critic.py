import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import gym
import time

class Actor(keras.Model):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.hidden1 = layers.Dense(64, activation='relu')
        self.hidden2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='softmax')
    
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.output_layer(x)
        return x

class Critic(keras.Model):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.hidden1 = layers.Dense(64, activation='relu')
        self.hidden2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)
    
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.output_layer(x)
        return x