import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import concurrent.futures
import copy
import pickle
from CNN_input import CNN
import gym
import numpy as np
from ES_classes import OpenES, CMAES, SimpleGA
from neural_diversity_net import NeuralDiverseNet
from rollout import fitness
import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(1)

## to run screenless:
##xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 main.py

inp_size = 648
action_size = 3
runs = ['CarRacing_run_001']



architecture = [inp_size, 128, 64,  action_size]

for run in runs:

    init_net = NeuralDiverseNet(architecture)

    init_params = init_net.get_params()
    fixed_weights = init_net.get_weights()

    cnn = CNN()

    print('trainable parameters:', len(init_params))

    EPOCHS = 4001
    popsize = 128
    cpus = 64

    
    with open('log_'+str(run)+'.txt', 'a') as outfile:
        outfile.write('trainable parameters: ' + str(len(init_params))+'\n')



    GAsolver = SimpleGA(len(init_params), popsize=512, sigma_init=1)

    def worker_fn(params):
        mean = 0
        for epi in range(1):
            net = copy.deepcopy(init_net)
            net.set_params(params)
            mean += fitness(net, copy.deepcopy(cnn))
        return mean/1


    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)


    for epoch in range(100):
        solutions = GAsolver.ask()
        with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
            fitlist = executor.map(worker_fn, [params for params in solutions])

        fitlist = list(fitlist)
        GAsolver.tell(fitlist)

        fit_arr = np.array(fitlist)

        print('epoch', epoch, 'mean', fit_arr.mean(), "best", fit_arr.max(), )
        with open('log_'+str(run)+'.txt', 'a') as outfile:
            outfile.write('GAepoch: ' + str(epoch)  + ' mean: ' + str(np.mean(fitlist)) + ' best: ' + str(np.max(fitlist)) + ' worst: ' + str(np.min(fitlist)) + ' std.: '  +str(np.std(fitlist)) + '\n')

        pop_mean_curve[epoch] = fit_arr.mean()
        best_sol_curve[epoch] = fit_arr.max()


    solver = CMAES(GAsolver.current_param(), popsize=128, sigma_init=.1)

    for epoch in range(100,EPOCHS):
        solutions = solver.ask()
        with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
            fitlist = executor.map(worker_fn, [params for params in solutions])

        fitlist = list(fitlist)
        solver.tell(fitlist)

        fit_arr = np.array(fitlist)

        print('epoch', epoch, 'mean', fit_arr.mean(), "best", fit_arr.max(), )
        with open('log_'+str(run)+'.txt', 'a') as outfile:
            outfile.write('epoch: ' + str(epoch)  + ' mean: ' + str(np.mean(fitlist)) + ' best: ' + str(np.max(fitlist)) + ' worst: ' + str(np.min(fitlist)) + ' std.: '  +str(np.std(fitlist)) + '\n')

        pop_mean_curve[epoch] = fit_arr.mean()
        best_sol_curve[epoch] = fit_arr.max()

        if (epoch + 1) % 50 == 0:
            with concurrent.futures.ProcessPoolExecutor(64) as executor:
                evaluations = executor.map(worker_fn, [solver.current_param() for i in range(64)])
            evaluations = list(evaluations)
            with open('log_'+str(run)+'.txt', 'a') as outfile:
                outfile.write('EVAL:   '+ ' mean: ' + str(np.mean(evaluations)) + ' best: ' + str(np.max(evaluations)) + ' worst: ' + str(np.min(evaluations)) + ' std.: '  +str(np.std(evaluations)) + '\n')
            eval_curve[epoch] = np.mean(evaluations)

        if (epoch + 1) % 50 == 0 or pop_mean_curve[epoch]>900 :
            print('saving..')
            pickle.dump((
                         solver,
                         copy.deepcopy(init_net),
                         copy.deepcopy(cnn),
                         pop_mean_curve,
                         best_sol_curve,
                         eval_curve
                         ), open(str(run)+'_' + str(len(init_params)) + '_' + str(epoch) + '_' + str(pop_mean_curve[epoch]) + '.pickle', 'wb'))

