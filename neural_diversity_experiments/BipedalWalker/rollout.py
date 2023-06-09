import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'                                     
os.environ['OMP_NUM_THREADS'] = '1'
import gym
import numpy as np
from neural_diversity_net import NeuralDiverseNet


def fitness(net: NeuralDiverseNet) -> float:
    env = gym.make("BipedalWalker-v3")

    obs = env.reset()
    done = False
    r_tot = 0
    t = 0
    r = 0
    while not done:
        action = net.forward(obs, r)
        obs, r, done, _ = env.step(action)
        r_tot += r
        t+=1
    env.close()
    return r_tot
