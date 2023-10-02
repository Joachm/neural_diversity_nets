import os
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['NUMEXPR_NUM_THREADS'] = '1'                                     
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import copy

class Neurons:
    def __init__(self, n_neurons):

        self.n_neurons = n_neurons

        self.neurons = np.zeros((n_neurons, ))
        self.ones = np.ones((n_neurons,1))
        self.hidden = np.zeros((n_neurons, 1))
        self.params = np.random.uniform(-1, 1, (n_neurons, 3,3))

    def neuron_fn(self, inputs):
        """
        inputs: (n_neurons, )
        """
        assert inputs.shape == (self.n_neurons,), inputs.shape
        
        inputs = inputs[:,np.newaxis]
        neurons = self.neurons
        params = self.params
         
        stacked = np.hstack((inputs, self.ones, self.hidden))[:,:,np.newaxis] #(n_neurons, 3)
        dot = np.tanh((self.params @ stacked).squeeze())
        x = copy.deepcopy(dot[:,0])
        self.hidden = dot[:,-1][:,np.newaxis]

        return x, dot

    def forward(self):
        pass  


class NeuralDiverseNet:
    def __init__(self, sizes):

        neurons = [Neurons(size) for size in sizes[:-1]]
        neurons.append(Neurons(sizes[-1]))
        self.neurons = neurons

        weights = [np.random.normal(0,.5, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.weights = weights

    def forward(self, x):
        """
        x: (n_in, )
        """
        neurons = self.neurons
        weights = self.weights

        pre, pre_dot = neurons[0].neuron_fn(x)

        for i, neuron in enumerate(neurons[1:]):
            send = pre @ weights[i]
            
            post, post_dot = neuron.neuron_fn(send)

            pre = post
            pre_dot = post_dot

        return post


    def get_params(self):
        p = np.vstack([neurons.params for neurons in self.neurons])

        return list(p.flatten())


    def set_params(self, flat_params):
        m = 0
        for neuron in self.neurons:
            a, b, c = neuron.params.shape
            neuron.params = flat_params[m:m + a * b * c].reshape(a, b, c)
            m += a * b * c


    def get_weights(self):
        return [w for w in self.weights]
