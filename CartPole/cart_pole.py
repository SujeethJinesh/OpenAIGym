import gym
import random
import numpy as np
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

env = gym.make('CartPole-v0')
env.reset()

learning_rate = 1e-3  # learning rate ie how quickly it detects something like outliers
goal_steps = 500  # desired number of frames we want to get to
score_requirement = 50  # since we're learning from random data to start with, we should start by selecting v good models
initial_games = 10000  # num games to start with/train on

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()  # allows you to sample random moves
            # pixel data, yes/no, game over, info
            observation, reward, done, info = env.step(action)
            if done:
                break

# some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

initial_population()
