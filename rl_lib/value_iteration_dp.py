import gym
import numpy as np

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

def init_trans_rewards(num_states, num_actions):

	return [[[0 for _ in range (num_states)] for _ in range(num_actions)] for _ in range(num_states)]

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



def sample_env(env):
	# Number of epsiodes
	for i_episode in range(20):
		observation = env.reset()

		# Timesteps in a given epsiode
		# Can use while True: till the very end of the episode
		for t in range(100):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			print (observation)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break;


def main():
	env = gym.make('FrozenLake-v0')
	num_states = env.observation_space.n
	num_actions = env.action_space.n
	P = init_model(num_states, num_actions)
	trans_counts = init_trans_counts(num_states, num_actions)
	trans_rewards = init_trans_rewards(num_states, num_actions)
	get_model_from_experience(trans_counts, trans_rewards)

if __name__ == '__main__':
	main();