import gym

env = gym.make('CartPole-v0')

for episode in range(1000):
	env.reset();
	for t in range(200):
		env.render();
		action = env.action_space.sample();
		observation, reward, done, info = env.step(action);
		if done:
			break;
