import gym
import random
import numpy as np
import tflearn

"""
input_data is input layer
droput - 20% --TODO
fully_connected layer
"""
from tflearn.layers.core import input_data, dropout, fully_connected

"""
regression for final layer
"""
from tflearn.layers.estimator import regression

"""
used to compare random movements based on learning
"""
from statistics import median, mean
from collections import Counter


LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def sample_run():
	for _ in range(5):

		state = env.reset()
		while True:
			env.render()

			action = env.action_space.sample()
			state, reward, done, info = env.step(action)

			if done:
				break


if __name__ == "__main__":
	main()
