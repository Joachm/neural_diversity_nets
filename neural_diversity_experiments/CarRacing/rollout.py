import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import gym
from gym import wrappers as w
from wrappers import ScaledFloatFrame
import numpy as np
from neural_diversity_net import NeuralDiverseNet
import torch
torch.set_num_threads(1)

def fitness(net: NeuralDiverseNet, cnn) -> float:
    env = gym.make('CarRacing-v0', verbose=0)
    env = w.ResizeObservation(env, 84)
    env = ScaledFloatFrame(env)
    obs = env.reset()
    obs = np.swapaxes(obs, 0,2)
    done = False
    r_tot = 0
    r = 0
    neg_count=0
    
    while not done:

        projection = cnn([obs])
        action = net.forward(projection, r)

        obs, r, done, _ = env.step(action)
        obs = np.swapaxes(obs, 0,2)

        r_tot += r

        ##Early stopping to save training time
        ##Don't use for evaluations!
        neg_count = neg_count+1 if r < 0.0 else 0
        if (done or neg_count > 30):
            break



    env.close()
    return r_tot
