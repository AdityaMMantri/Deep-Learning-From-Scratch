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