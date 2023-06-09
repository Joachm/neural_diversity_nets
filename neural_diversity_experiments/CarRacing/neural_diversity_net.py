import os
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['NUMEXPR_NUM_THREADS'] = '1'                                     
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import copy
from scipy.special import softmax


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class Neurons:
    def __init__(self, n_neurons, last=False):
        """
        A single layer of linear projection followed by the complex neuron activation function
        """
        self.n_neurons = n_neurons

        self.neurons = np.zeros((n_neurons, ))
        self.ones = np.ones((n_neurons,1))
        self.hidden = np.zeros((n_neurons, 1))
        self.params = np.random.uniform(-1, 1, (n_neurons, 3,3))
        self.last = last

    def neuron_fn(self, inputs, r):
        """
        inputs: (n_neurons, )
        """
        assert inputs.shape == (self.n_neurons,), inputs.shape
        
        inputs = inputs[:,np.newaxis]
        neurons = self.neurons
        params = self.params
         
        stacked = np.hstack((inputs, self.ones, self.hidden))[:,:,np.newaxis] #(n_neurons, 3)
        
        if not self.last:
            dot = np.tanh((self.params @ stacked).squeeze())
                
            x = copy.deepcopy(dot[:,0])
            self.hidden = dot[:,-1][:,np.newaxis]

            return x, dot
        if self.last:
            dot = (self.params @ stacked).squeeze()
                
            x = copy.deepcopy(dot[:,0])
            self.hidden = np.tanh(copy.deepcopy(dot[:,-1][:,np.newaxis]))
        
            ##Specific output for CarRacing
            x[0] = np.tanh(x[0])
            x[1:] = sigmoid(x[1:])

            return x, np.tanh(dot)



    def forward(self):
        pass  


class NeuralDiverseNet:
    def __init__(self, sizes):
        """
        A fully connected network of ComplexLayer

        sizes: [input_size, hid_1, ..., output_size]
        add_identity_input_layer: bool, whether to add an input layer with an identity weight matrix
        """
        neurons = [Neurons(size) for size in sizes[:-1]]
        neurons.append(Neurons(sizes[-1], last=True))
        self.neurons = neurons

        weights = [np.random.normal(0,.5, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.weights = weights

    def forward(self, x, r):
        """
        x: (n_in, )
        """

        neurons = self.neurons
        weights = self.weights

        pre, pre_dot = neurons[0].neuron_fn(x, r)

        for i, neuron in enumerate(neurons[1:]):
            send = pre @ weights[i]
            
            post, post_dot = neuron.neuron_fn(send, r)

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
