'''
This file contains Activation Functions used in the neural network.
Activation functions introduce NON-LINEARITY into the network.
Without activations,multiple Linear layers collapse into a single Linear transformation.

1. Apply non-linear transformations
2. Help the network learn complex patterns
3. Store inputs needed for backpropagation
4. Compute gradients during backward pass

FORWARD PASS
Takes output from a Linear layer
and applies a non-linear transformation.

BACKWARD PASS
Computes gradients of the activation function and passes gradients backward.
''' 

import numpy as np
from module import Module

class Sigmoid(Module):

    def forward(self,inputs):
        # computes activated output during forward propagation
        output=1/(1+np.exp(-inputs)) # Sigmoid activation
        self.output=output # caching for backprop
        return output
    
    def backward(self,gradient_outputs):
        # computes gradients during backpropagation, baiscaillly will be used to calcuate the gradient for previou layer
        sigmoid_gradient=self.output*(1-self.output) # Derivative of sigmoid
        gradient_inputs=gradient_outputs*sigmoid_gradient # Apply chain rule
        return gradient_inputs
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []

class Tanh(Module):

    def forward(self,inputs):
        output=np.tanh(inputs)
        self.output=output
        return output
    
    def backward(self,gradient_outputs):
        tanh_gradient=1-(self.output**2)
        gradient_inputs=gradient_outputs*tanh_gradient
        return gradient_inputs
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []

class ReLU(Module):
    
    def forward(self,inputs):
        self.inputs=inputs # caching inputs becuase we will use inputs in backprop
        output=np.maximum(0,inputs)
        return output
    def backward(self,gradient_outputs):
        gradient_inputs=gradient_outputs.copy() # Copy incoming gradients
        gradient_inputs[self.inputs <= 0] = 0
        return gradient_inputs
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []

