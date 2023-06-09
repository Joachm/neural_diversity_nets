import gym
import numpy as np

# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # This undoes the memory optimization, use with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


