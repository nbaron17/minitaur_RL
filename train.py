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


def train(env):
    model = PPO('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    # model = PPO.load('models/PPO_300000.zip', env=env, device='cuda', tensorboard_log=log_dir)
    TIMESTEPS = 50000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/PPO_{TIMESTEPS*iters}")


def main():
    env = DummyVecEnv([lambda: Monitor(gym.make("GymV21Environment-v0", env_id="MinitaurBulletEnv-v0"))])
    # check_env(gym.make('MinitaurBulletEnv-v0'))
    train(env)


if __name__ == '__main__':
    main()
