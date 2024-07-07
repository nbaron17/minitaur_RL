import cv2
import numpy as np
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

    # Initialize video writer
    frame_width = 600  # Define width according to the environment's render size
    frame_height = 400  # Define height according to the environment's render size
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
    video_writer = cv2.VideoWriter('trained_dog.avi', fourcc, fps, (frame_width, frame_height))

    num_frames = 500  # Number of frames to record

    for i in range(num_frames):
        action, _ = model.predict(obs)
        obs, reward, done, info, _ = env.step(action)

        # Render the environment
        frame = env.render()

        # Convert the frame to BGR format (if needed)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize the frame to the specified dimensions
        resized_frame = cv2.resize(bgr_frame, (frame_width, frame_height))

        # Write the frame to the video file
        video_writer.write(resized_frame)
        print(i)
        if done:
            env.reset()

    # Release the video writer and close the Gym environment
    video_writer.release()
    env.close()


def main():
    env = gym.make("GymV21Environment-v0", env_id="MinitaurBulletEnv-v0")
    test(env, path_to_model="models/PPO_300000.zip")


if __name__ == '__main__':
    main()
