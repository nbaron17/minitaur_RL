import gymnasium as gym
import pybullet_envs


import gymnasium as gym

env = gym.make("GymV21Environment-v0", env_id="MinitaurBulletEnv-v0")
# env_name = 'MinitaurBulletEnv-v0'
# env = gym.make(env_name, render="human")
obs = env.reset()[0]
done = False
while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info, _ = env.step(action)
env.close()
