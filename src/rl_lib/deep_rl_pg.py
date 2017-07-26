import gym
import numpy as np
from math import sqrt

"""
Using deep learning and policy gradients on Gym's game of Pong

At a low level, given the current screen in game play with certain pixel
arrangement, we need to know if the paddle should move UP or DOWN.

A policy network defines the agent. The screen pixels are fed to this policy 
network which decides if the paddle should move UP or DOWN (a probability)

Give a sense of movement of the ball to the policy network by preprocessing
input data. eg: taking difference between before and after frame

Credit assignment problem: so many variations in movement, not sure what caused
a good or bad move. (the latest move? or a move sometime ago?)

Generally, in supervised learning we want to maximize the log probabilities
of all the correct decisions given an input frame. Since we do not know 
the correct answer we take the log probs of all the actions we take till
the end of the game, and at the end multiply those log probs with a scalar
(+1 if we won, -1 if we lost). Accordingly maximizing this total sum will
encourage good actions and discourage bad actions. Perform gradient ascent.

"""

"""
General procedure:
- Get a state - which is in the frame of the atari game
- Preprocess the frame - resizing it to something smaller
- Feed this into a 2 layer neural network
- Spits out the probability of going UP, sample from this distritbution
- After an episode (reaching 21 points), compute gradients
	- Our result is a sigmoid function
	- During backprop we need the local gradients
	- one being, dL/df where f is linear combination of all the ReLU outputs
	- taking the derivative (http://cs231n.github.io/neural-networks-2/#losses)
	- Above is based on gradient ascent. check logistic regression: https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf
	- But we dont know the correct label unlike the supervised learning example above
	- so we consider the action we took as correct in this RL problem
	- So a frame is inputed into the NN, out comes a probability
	- This has an associated loss, we can get the rate of change of this loss wrt f
	- using "correct" answer as what we ended up sampling
- after 10 episodes (minibatch SGD) sum the gradients and move in gradient direction
- want to use policy gradients
	- at the end of an episode (when someone reaches 21 points)
	- we encourage all actions that let us win, or discourage actions that made us lose

"""

class PongAI(object):

	def __init__(self):
		self.env = gym.make('Pong-v0')
		self.image_dim = 80 * 80 # pre processed image size
		self.num_hidden_layer = 200 # number of neurons in single hidden layer
		self.batch_size = 10 # mini batch for stochastic gradient descent
		self.gamma = 0.99 # discount factor for reward function
		self.decay_rate = 0.99 # RMSprop decay rate, as per Hinton recommendation
		self.learning_rate = 0.001 # alpha - during gradient descent, Hinton recommendation


		# Xavier inits of weights
		# Divide by image_dim because 
		# Xavier init worries about weights going into a single neuron
		# here 6400 go into a neuron, if more weights each has lesser weight
		# if less weights (< 6400) then each has higher weights
		# similar you do that for each of the nerons in the hidden layer
		# 200 * 6400
		# weights for second layer
		# 200 * 1
		self.weights = {
			'w1': np.random.randn(self.num_hidden_layer, self.image_dim) * sqrt(2.0/self.image_dim),
			'w2': np.random.randn(self.num_hidden_layer, 1) * sqrt(2.0/self.num_hidden_layer)
		}

		# RMSprop inits. Used for optimizing gradient descent (http://ruder.io/optimizing-gradient-descent/index.html#rmsprop)
		self.expectation_g_squared = {}
		self.g_dict = {}
		for layer_name in self.weights.keys():
			self.expectation_g_squared[layer_name] = np.zeros_like(self.weights[layer_name]) # return array of this type w all zeros
			self.g_dict[layer_name] = np.zeros_like(self.weights[layer_name])
	
	def sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-x))

	# prepro image passed in is a 1 * 6400 unrolled image
	def feedforward(self, prepro_image):
		h = np.dot(prepro_image, self.weights['w1'].T)
		
		h[h <= 0] = 0 # ReLU non linear function
		# h is a 1 * 200 vector - output from each neuron
		# 1 * 200 * 200 * 1 = 1 element
		p = np.dot(h, self.weights['w2'])
		return self.sigmoid (p), h


	# Process the image
	# Take difference in the current and previous state frames
	# return the the difference as input for the neural network
	def preprocess(self, current_state, prev_state):
		""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
		current_state = current_state[35:195] # crop
		current_state = current_state[::2,::2,:] # downsample by factor of 2 - halving the resolution of the image (taking alternate pixels)
		current_state = current_state[:,:,0] # removing RGB colors
		current_state[current_state == 144] = 0 # erase background (background type 1)
		current_state[current_state == 109] = 0 # erase background (background type 2)
		current_state[current_state != 0] = 1 # everything else (paddles, ball) just set to 1

		# 1 * 6400
		current_state = current_state.astype(np.float).ravel()

		if prev_state is not None:
			# Difference in the frames fed as input to the neural network
			# Difference taken to instill some concept of motion of the ball
			nn_input_state = current_state - prev_state
		else:
			nn_input_state = np.zeros(self.image_dim) # 1 * 6400 

		prev_state = current_state
		return nn_input_state, prev_state
	
	def choose_action(self, up_probability):

		p = np.random.random()

		if p < up_probability:
			return 2 # go up
		else:
			return 3 # go down

	def discount_rewards(self, rewards):

		discounted_rewards = np.zeros_like(rewards)
		running_reward = 0
		gamma = self.gamma

		# discounting rewards in the future
		# G1 = r1 + gamma*r2 + gamma^2 *r3 + ... + gamma^(n-1) * rn
		# Gn-1 = rn-1 + gamma * rn
		# Gn = rn
		for t in reversed(range(rewards.size)):

			if rewards[t] != 0:
				running_reward = 0 #TODO - ??

			running_reward = gamma * running_reward + rewards[t]
			discounted_rewards[t] = running_reward

		# reward pre processing
		discounted_rewards -= np.mean(discounted_rewards)
		discounted_rewards /= np.std(discounted_rewards)
		return discounted_rewards

	# BACKPROP!
	# computes
	# dL/dw1, dL/dw2
	# gradients of loss wrt set of first weights and second weights
	# essentially thing needed to govern how distribution will shift
	def compute_gradients(self, episode_gradient_log_ps_discounted, episode_hidden_layer_values, episode_observations):

		# let number of frames before end be F
		# then first arg is F * 1
		# second arg is F * 200 (because 200 hidden layers) for each frame as input into the NN
		# results in a 200 * 1 matrix
		# Each row in the 200 * 1 represents the sum of all gradients (each f for each diff input into nn
		# ) wrt first weight. then second weight and so on
		# Based on dL/dw2 = (dL/df) * (df/dw2)
		dL_dw2 = np.dot(episode_hidden_layer_values.T, episode_gradient_log_ps_discounted).ravel()

		# gradient of loss wrt first set of weights
		# draw computational graph and backprop
		# we have vector F * 1 (of all dL/df)
		# we have vector 200 * 1 (weights in second layer)
		# by computing backprop by hand we can see we need F * 200
		# ie: for each episode we need a row of 200 entries such that each entry
		# is dL/df * wi. Because wi = df/d(relu(f1)))

		# whole thing is basically (ignore bad notation)
		# dL/w1 = dL/df * df/dRelu(f1) * dRelu(f1)/df1 * df1/d(wx) * d(wx)/dw1
		delta_l2 = np.outer(episode_gradient_log_ps_discounted, self.weights['2']) # F * 200 matrix
		delta_l2[delta_l2 <= 0] = 0 # ReLU, equivalent to dRelu(f1)/df1 = 1 or 0 TODO - ??

		# After getting till here, idea is that each gradient has to be multiplied with each
		# input (because d(wx)/dw = x)
		# And all gradients wrt same thing are added across frames
		# in other words dL1/dw1 + dL2/dw1, wrt is same but L1 is cost for first frame, L2 is for second
		# This is similar to abaove idea of summing all gridients for minibatch SGD
		dL_dw1 = np.dot(delta_l2.T, episode_observations) # the dwx/dw is just x (the input)
		return {
			'1': dL_dw1,
			'2': dL_dw2
		}

	# Using RMSprop
	# Based on http://ruder.io/optimizing-gradient-descent/index.html#rmsprop
	def update_weights(self):
		epsilon = 1e-5

		# Updating 1 layer at a time
		for layer_name in self.weights:
			self.expectation_g_squared[layer_name] = self.decay_rate * self.expectation_g_squared[layer_name] + (1 - self.decay_rate) * self.g_dict[layer_name] ** 2

			# Update!!! Gradient ASCENT
			self.weights[layer_name] = self.weights[layer_name] + (self.learning_rate) * g_dict[layer_name] / (sqrt(expectation_g_squared[layer_name] + epsilon))
			# reset for next batch
			self.g_dict[layer_name] = np.zeros_like(self.weights[layer_name])

	 

	def start_learning(self):
		
		episode_number = 0
		reward_sum = 0
		current_reward = None
		prev_state = None
		current_state = self.env.reset()

		# Collecting data as episode proceeds
		episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
		
		while True:
			self.env.render();
			nn_input_state, prev_state = self.preprocess(current_state, prev_state)

			# keeping track of observations passed to NN
			episode_observations.append(nn_input_state)

			up_probability, hidden_layer_output = self.feedforward(nn_input_state)

			# keeping track of hidden layer output after ReLU
			episode_hidden_layer_values.append(hidden_layer_output)

			# Should we move up or down
			action = self.choose_action(up_probability)

			# for backprop purposes we need dL/df. f is linear comb of ReLU outputs
			# L is the sigmoid function result (the probability)
			# turns out that dL/df = y - sigmoid(f)
			# We dont have correct answer, y. So we consider whatever we did in "action"
			# to be correct
			# here let dL/df = loss_grad_f
			# consider "correct" answers here as fake labels of classification
			# if action is up, the classified as 1 else 0 (treated as a binary classification problem)
			# 1 is class of up, 0 is of down
			fake_label = 1 if action == 2 else 0

			# as per formula of gradient
			loss_grad_f = fake_label - up_probability
			episode_gradient_log_ps.append(loss_grad_f)

			
			current_state, reward, done, info = self.env.step(action)

			reward_sum += reward

			# keep track of rewards
			episode_rewards.append(reward)

			# done is considered when a player reaches 21 points
			# a = [11,12,13]
			# np.vstack(a) = [[11], [12], [13]]
			if done:
				episode_number += 1
				episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
				episode_observations = np.vstack(episode_observations)
				episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
				episode_rewards = np.vstack(episode_rewards)

				# we want to weight actions closer to the end of the episode
				# more heavily than ealier ones because it is more likely that
				# later actions affected outcome of the episode. Use discounting.
				episode_discounted_rewards = self.discount_rewards(episode_rewards)

				# Starting off policy gradients
				# f(x) = reward function
				# dL/dw
				# we want f(x)*(dL/dw)
				# with below we have dL/df. We get dL/dw using chain rule and 
				# simply chaining the multiplication
				# element wise multiplication
				episode_gradient_log_ps_discounted = episode_gradient_log_ps * episode_discounted_rewards
				gradients = compute_gradients(episode_gradient_log_ps_discounted, episode_hidden_layer_values, episode_observations)

				# at this point we have computed gradient taking into account many frames
				# to get the loss, rate of change etc
				# keep adding these up and when batch multiple hits
				# then update in one
				# Sum the gradients and update at once when the batch hits
				for layer in gradients:
					self.g_dict[layer] += gradients[layer]

				if episode_number % self.batch_size == 0:
					update_weights()

				# Reset everything for the next game
				episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

				current_state = self.env.reset()
				prev_state = None

#agent = PongAI()

def temp_sim():
	env = gym.make('Pong-v0')

	env.reset()

	while True:
		env.render()

		state, reward, done, _ = env.step(env.action_space.sample())
		if reward !=0:
			print(reward);

		if done:
			print("done!!!!")
			break;

temp_sim()

#p = PongAI()
#p.start_learning()












