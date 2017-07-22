import gym
import numpy as np

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
encourage good actions and discourage bad actions.

"""

"""
General procedure:
- Get a state - which is in the frame of the atari game
- Preprocess the frame - resizing it to something smaller
- Feed this into a 2 layer neural network
- Spits out the probability of going UP
-  

"""

class PongAI(object):

	def __init__(self):
		self.env = gym.make('Pong-v0')

	# Credit to Karpathy's pg-pong.py
	def preprocess(self, image):
		""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
		image = image[35:195] # crop
		image = image[::2,::2,0] # downsample by factor of 2
		image[image == 144] = 0 # erase background (background type 1)
		image[image == 109] = 0 # erase background (background type 2)
		image[image != 0] = 1 # everything else (paddles, ball) just set to 1
		return image.astype(np.float).ravel()
















