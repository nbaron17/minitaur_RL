import gymnasium as gym
import pybullet_envs
from stable_baselines3 import PPO
import os
import sys
import time
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def test(env, path_to_model):
    model = PPO.load(path_to_model, env=env)

    # env.reset()
    obs = env.reset()[0]
    done = False
    extra_steps = 5000
    while True:
        action, _ = model.predict(obs)
        print(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            extra_steps -= 1
            if extra_steps < 0:
                break


def main():
    env = gym.make("GymV21Environment-v0", env_id="MinitaurBulletEnv-v0")
    test(env, path_to_model="models/PPO_300000.zip")


if __name__ == '__main__':
    main()
