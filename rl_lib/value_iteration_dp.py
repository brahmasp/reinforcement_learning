import gym
import numpy as np
import time
from lake_envs import *

"""
General idea of DP solution

1. Start off with a random model of the environment
2. Improve the model as and when experience goes on
	2.1 Mark terminal states as absorbing state. In other words, prob of returning 
	to itself is 1 and reward 0
	2.2 Update counts and rewards each episode based on all visited states/actions/rewards etc
3. When finally converges - average rewards, average transition probabilities
then use this to estimate the value function
4. Perform control accordingly
"""

"""
Initing the model of the environment
For a particular state and action, we get a list of tuples
list because after taking a particular action from a state multiple
resulting states are possible

(transition prob, next state, reward, done)

"""
def init_model(num_states, num_actions):

	return [[[(1.0/num_states, i, 0, False) for i in range(num_states)] for _ in range(num_actions)] for _ in range(num_states)]

"""
Building the overall model involves knowing
transition prob and rewards for each transition

trans_counts[state][action][state_end] gives number of times from state to 
state_end via taking action

Init all transition counts are zero
"""

def init_trans_counts(num_states, num_actions):
	
	return [[[0 for _ in range(num_states)] for _ in range(num_actions)] for _ in range(num_states)]

"""
trans_rewards[state][action][next_state] is reward received for the transition 
from state to next_state after taking action 

"""
def init_trans_rewards(num_states, num_actions):

	return [[[0 for _ in range (num_states)] for _ in range(num_actions)] for _ in range(num_states)]


"""
Get overall model (P) from the transition counts and rewards

"""
def get_model_from_experience(trans_counts, trans_rewards):

	# Create empty model - for a given pair of state and action there is a
	# corresponding list
	num_states = len(trans_counts)
	num_actions = len(trans_counts[0])
	P = [[[] for _ in range(num_actions)] for _ in range(num_states)]

	for state in range(num_states):
		for action in range(num_actions):
			total_count = sum(trans_counts[state][action])
			for next_state in range(num_states):
				count = trans_counts[state][action][next_state]
				reward = trans_rewards[state][action][next_state]
				t = ();
				if total_count != 0:
					t = (float(count) / total_count, next_state, reward, False)
				else:
					t = (1.0 / num_states, next_state, 0, False)
				P[state][action].append(t);
	return P;

"""
As experience/episodes go on, want to update trans_counts, trans_rewards
as needed.
These updates to trans_counts, trans_rewards will be used to form 
final model using "get_model_from_experience"

hist is a list  [state, action, reward, next_state, done]
This list is per episode basis

index corresponds
0 to state
1 to action
and so on
"""
def update_experience(trans_counts, trans_rewards, hist):

	num_actions = len(trans_counts[0])
	for state, action, reward, next_state, done in hist:

		# Work for the terminal state if we have reached it
		# otherwise proceed as normal

		# This type of transition was experienced
		trans_counts[state][action][next_state] += 1;
		curr_reward = trans_rewards[state][action][next_state]
		count = trans_counts[state][action][next_state]

		# Incremental mean
		# A <- A + k * (reward - A)
		# where k = 1 / count
		trans_rewards[state][action][next_state] = curr_reward + (1.0 / count)*(reward - curr_reward)

		if done:
			# Terminal state can be success or failure
			# Set the reward of that state accordingly
			for a in range(num_actions):
				trans_counts[next_state][a][next_state] = 1
				trans_rewards[next_state][a][next_state] = 0

	return trans_counts, trans_rewards

def construct_model(env, num_states, num_actions, num_episodes = 50000):

	P = init_model(num_states, num_actions)
	trans_counts = init_trans_counts(num_states, num_actions)
	trans_rewards = init_trans_rewards(num_states, num_actions)

	term_states = []
	for i_episode in range(num_episodes):

		# some init state 
		state = env.reset()

		# list of all individual experiences in a given episode
		hist = []
		while True:
			sub_hist = []
			env.render()

			action = env.action_space.sample()

			# init state
			sub_hist.append(state)

			# action chosen
			sub_hist.append(action)

			state, reward, done, _ = env.step(action)
			# reward received
			sub_hist.append(reward)

			# next_state
			sub_hist.append(state)

			# terminal state or not
			sub_hist.append(done)

			# keep track of this experience in the history list
			hist.append(sub_hist)
			if done:
				term_states.append(state)
				print "Finished episode {} and state {}".format(i_episode + 1, state);
				break;

		# for each episode, we collect the history
		# consume this history and keep track of it all to use when building
		# model of environment
		trans_counts, trans_rewards = update_experience(trans_counts, trans_rewards, hist)

	P = get_model_from_experience(trans_counts, trans_rewards)

	return value_iterate(P, num_states, num_actions)

"""
After value iteration converges, time to extract the policy

"""
def policy_extraction(P, num_states, num_actions, V, policy, discount = 0.9):

	for s in range(num_states):
		q = []
		for a in range(num_actions):
			next_states = P[s][a]
			value = 0
			for next_s in next_states:

				# relevant only if possible to go this next state
				# so probability is not zero
				# Given an action how much value does it give
				# collect all these values
				# find the max value. thats where we want to go
				# index of max corresponds to desired action
				value += next_s[0] * (next_s[2] + discount * V[next_s[1]])
			q.append(value)

		policy[s] = np.argmax(q)
	return policy
			

"""
Performing value iteration
"""
def value_iterate(P, num_states, num_actions, discount = 0.9):

	V = np.zeros((1, num_states))[0]
	policy = [-1 for _ in range(num_states)]

	for _ in range (2500):
		for s in range(num_states):
			v = V[s];
			q = []
			for a in range(num_actions):
				value = 0
				next_states = P[s][a]
				for next_s in next_states:
					# Expected value of state
					# prob * (reward + value(s'))
					value += next_s[0] * (next_s[2] + discount * V[next_s[1]])
				q.append(value)
			V[s] = max(q);

	policy = policy_extraction(P, num_states, num_actions, V, policy)
	# V = np.reshape(V, (8,8))
	# policy = np.reshape(policy, (8,8))
	# print("After value iteration \n{}".format(V))
	# print("Optimal policy \n{}".format(policy))
	
	return V, policy

		
"""
Sample run to test out any policy (optimal, random etc)
"""
def sample_env(file, env, V, policy, random = False):

	# Timesteps in a given epsiode
	# Can use while True: till the very end of the episode
	goal_reached = 0;
	number_of_episodes = 100

	for i_episode in range(number_of_episodes):
		state = env.reset();
		while True:
			env.render()
			if random:
				action = env.action_space.sample()
			else:
				action = policy[state]
			
			state, reward, done, _ = env.step(action)
			if done:
				if reward != 0:
					goal_reached += 1
				print("episode terminated, reached state {}".format(state))
				break


	result_string = "Goal reached: {}%".format(float(goal_reached * 100.0) / number_of_episodes)
	print(result_string)
"""
0 -> left
1 -> down
2 -> right
3 -> up
"""
def main():
	# Success rates in order
	# 100%
	# 100%
	# ~75%
	# ~85%
	env_list = ['Deterministic-4x4-FrozenLake-v0', 'Stochastic-4x4-FrozenLake-v0', 'Deterministic-8x8-FrozenLake-v0', 'FrozenLake8x8-v0']
	
	for env_item in env_list:
		env = gym.make(env_item)
		num_states = env.observation_space.n
		num_actions = env.action_space.n
		V, policy = construct_model(env, num_states, num_actions)
		sample_env(file, env, V, policy)
		# uncomment below to for random action
		#sample_env(file, env, V, policy, True)


if __name__ == '__main__':
	main();