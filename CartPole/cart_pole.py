import gym
import random
import numpy as np
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
from gym import wrappers
from ../config/keys import api_key

env = gym.make('CartPole-v0')
env.reset()

learning_rate = 1e-3  # learning rate ie how quickly it detects something like outliers
goal_steps = 500  # desired number of frames we want to get to
score_requirement = 70  # since we're learning from random data to start with, we should start by selecting v good models
initial_games = 10000  # num games to start with/train on

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = env.action_space.sample()  # sampling from the action space
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:  # one hot encoding the actions taken
                if data[1] == 1:  # taking out the action from the game memory
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])  # putting in the prev. observation and output?
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))
    print('Max accepted score: ', max(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    # Prevent overfitting by adjusting epochs
    model.fit({'input': X}, {'targets': y}, n_epoch=7, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


training_data = initial_population()
model = train_model(training_data=training_data)

scores = []
choices = []

env = wrappers.Monitor(env, './cartpole-experiment-1', force=True)
for each_game in range(100):
    score = 0
    game_memory = []
    prev_observations = []
    env.reset()
    for _ in range(goal_steps):
        # env.render()
        if len(prev_observations) == 0:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(prev_observations.reshape(-1, len(prev_observations), 1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_observations = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)

print('Average Score', sum(scores)/len(scores))

env.close()

# gym.upload('./cartpole-experiment-1', api_key=api_key)

model.save('cartpole-experiment-1.model')
