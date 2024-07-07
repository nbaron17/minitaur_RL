# Minitaur RL via Stable Baselines 3
Train minitaur quadruped to walk via RL using stable baselines 3. PPO is the pretrained model here.

## Installation
Run:

    pip install -r requirements.txt

## Usage
To run a sample (just random actions), run:

    python main.py

In order to render the simulation, the usual render_mode="human" argument when calling gym.make() doesn't work here.
A work around is to modify the render argument in 
venv/lib/python3.10/site-packages/pybullet_envs/bullet/minitaur_gym_env.py (line 69) from False to True.

Note that if the following error is generated: AttributeError: `np.string_` was removed in the NumPy 2.0 release. 
Use `np.bytes_` instead.. Did you mean: 'strings'? - try the solution given here: 
https://github.com/tensorflow/tensorboard/issues/6874

To train a model, run:

    python train.py
