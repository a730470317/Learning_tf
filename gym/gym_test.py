import gym

def run_cat():
    env = gym.make("Taxi-v2")
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

def run_CartPole():
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action

def run_Acrobot_v1():
    env = gym.make('Acrobot-v1')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action

print("gym version: = ", gym.__version__)
# run_cat()
# run_CartPole()
run_Acrobot_v1()