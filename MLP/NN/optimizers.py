'''
This file contains Optimizers used to update the trainable parameters of the neural network.

1. Access model parameters and gradients
2. Update weights and biases
3. Help reduce the loss over time

HOW IT WORKS
After backpropagation:
 - every layer contains gradients

The optimizer:
 - loops through all parameters
 - updates them using gradients
''' 

import numpy as np
from sequential import Sequential

class SGD:
    def __init__(self,learning_rate):
        if learning_rate<=0:
            raise ValueError("Learning rate must be positive")
        self.learning_rate=learning_rate

    def update_step(self,model:Sequential): # again type hinting is done beacuse sequential object will be passed.
        parameters=model.parameters()
        gradients=model.gradients()

        for parameters,gradients in zip(parameters,gradients): # zip is to form pairs (W1,dW1) etc.
            parameters-=self.learning_rate*gradients # update rule

class Mometum:
    def __init__(self,learning_rate,beta):
        if learning_rate<=0:
            raise ValueError("Learning rate must be positive")
        if not(0<=beta<=1):
            raise ValueError("Momentum Coefficent must be between 0 and 1")
        self.learning_rate=learning_rate
        self.beta=beta
        self.velocities=None
    
    def update_step(self,model:Sequential):
        parameters=model.parameters()
        gradients=model.gradients()

        if self.velocities is None:
            self.velocities=[]
            for parameter in parameters:
                velocity=np.zeros_like(parameter)
                self.velocities.append(velocity)
        for i,(parameter,gradient) in enumerate(zip(parameters,gradients)):
                self.velocities[i] = (self.beta * self.velocities[i]+ (1 - self.beta) * gradient)
                parameter -= (self.learning_rate* self.velocities[i])
# momentum is not fully written by me just fucking shit optimizer ಠ╭╮ಠ