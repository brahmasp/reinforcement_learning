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
- Spits out the probability of going UP
- After an episode (reaching 21 points), compute gradients
- after 10 episodes (minibatch SGD) sum the gradients and move in gradient direction

"""

class PongAI(object):

	def __init__(self):
		self.env = gym.make('Pong-v0')
		self.image_dim = 80 * 80 # pre processed image size
		self.num_hidden_layer = 200 # number of neurons in single hidden layer
		self.batch_size = 10 # mini batch for stochastic gradient descent
		self.gamma = 0.99 # discount factor for reward function
		self.decay_rate = 0.99 # RMSprop decay rate
		self.learning_rate = 0.001 # alpha - during gradient descent

		# Xavier inits of weights
		# Divide by image_dim because 
		# Xavier init worries about weights going into a single neuron
		# here 6400 go into a neuron, if more weights each has lesser weight
		# if less weights (< 6400) then each has higher weights
		# similar you do that for each of the nerons in the hidden layer
		# 200 * 6400
		self.w1 = np.random.randn(self.num_hidden_layer, self.image_dim) * sqrt(2.0/self.image_dim)
		
		# weights for second layer
		# 200 * 1
		self.w2 = np.random.randn(self.num_hidden_layer, 1) * sqrt(2.0/self.num_hidden_layer)

		# RMSprop inits. Used for optimizing gradient descent (http://ruder.io/optimizing-gradient-descent/index.html#rmsprop)
		



	def sigmoid(x):
		return 1.0 / (1.0 + np.exp(-x))

	# prepro image passed in is a 1 * 6400 unrolled image
	def feedforward(self, prepro_image):
		h = np.dot(prepro_image, self.w1)
		h[h <= 0] = 0 # ReLU non linear function
		# h is a 1 * 200 vector - output from each neuron
		# 1 * 200 * 200 * 1 = 1 element
		p = np.dot(h, self.w2)
		return sigmoid (p)


	# Credit to Karpathy's pg-pong.py
	def preprocess(self, image):
		""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
		image = image[35:195] # crop
		image = image[::2,::2,0] # downsample by factor of 2
		image[image == 144] = 0 # erase background (background type 1)
		image[image == 109] = 0 # erase background (background type 2)
		image[image != 0] = 1 # everything else (paddles, ball) just set to 1
		return image.astype(np.float).ravel()



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
			break;

#temp_sim()












