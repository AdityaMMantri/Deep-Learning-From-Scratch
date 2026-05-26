'''
This file defines the Sequential Model,which stores and manages all layers of the neural network.
The network is treated as an ordered sequence of layers.
1. Store all layers of the neural network
2. Handle forward propagation
3. Handle backward propagation
4. Collect parameters and gradients from all layers

FORWARD PASS
Input is passed layer-by-layer:
     Input → Linear → ReLU → Linear → Output
Each layer receives output from the previous layer.

BACKWARD PASS
Gradients flow in reverse order:
    Output → Linear → ReLU → Linear → Input

Each layer computes gradients and passes them backward.
'''

from NN.module import Module
from typing import Tuple

# tensorflow style API is copied (～￣▽￣)～ although pytorch is better ;)
class Sequential(Module):

    def __init__(self, *layers:Module):
        # Store all layers of the network in order.
        # * -> unpacking/packing opertor-> will take all the layer objects and store them in a tuple 
        # to convert to a single argumment because a NN can have many layers meaning any possible number of arguments
        # Type hinting is done might not be required but just in case.
        self.layers: Tuple[Module, ...] = layers

    def forward(self,inputs):
            # dynamically calls:
            # Linear.forward()
            # ReLU.forward()
            # Sigmoid.forward()
            # etc depending on object type
        for layer in self.layers:
            inputs=layer.forward(inputs)
        return inputs
    
    def backward(self,gradient):
        for layer in reversed(self.layers):
            gradient=layer.backward(gradient)
        return gradient
    
    def parameters(self):
        # Sequential combines ALL layer parameters into one list.
        # Collect all trainable parameters from every layer.
        params=[] # will coontain all paramters [W1,b1,W2,b2.....]
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    
    def gradients(self):
        # Collect all parameter gradients from every layer.
        # exampel-> [dW1, db1, dW2, db2]
        # extend function beacuse we want each element to be added individually
        grads=[]
        for layer in self.layers:
            grads.extend(layer.gradients())
        return grads
