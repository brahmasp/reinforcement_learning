import gym

env = gym.make('CartPole-v0')
observation_n = env.reset()

while True:
  action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
  observation_n, reward_n, done_n, info = env.step(action_n)
  env.render()