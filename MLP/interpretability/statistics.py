"""
This module contains statistical analysis utilities for
studying neural network behavior during and after training.

The goal is to measure and summarize important properties
of weights, activations, and gradients.

By analyzing statistics, we can detect:
- dead neurons
- activation sparsity
- exploding or vanishing values
- unstable training behavior
- representation collapse

This file may include:
- mean and standard deviation analysis
- sparsity measurements
- neuron activity statistics
- gradient statistics
- layer-wise summaries
"""
import numpy as np
import matplotlib.pyplot as plt

from NN.sequential import Sequential

def parameter_count(model:Sequential):
    '''
    Counts the total number of trainable parameters
    inside the neural network.

    Trainable parameters include:
        - weights
        - biases
    '''
    total_parameters=0
    for layer in model.layers:
        for parameter in layer.parameters():
            total_parameters+=parameter.size
    return total_parameters

def model_summary(model:Sequential):
    '''
    Displays a TensorFlow-style summary
    of the neural network architecture.
    '''
    print("\nModel Summary")
    print("="*95)
    # TABLE HEADER
    print(
        f"{'Layer':<20}" # 20 character width andleft aligned
        f"{'Weights Shape':<20}"
        f"{'Bias Shape':<20}"
        f"{'Param #':<15}"
    )
    print("=" * 95)
    for i,layer in enumerate(model.layers):
        '''
        model.layers = [Linear(), ReLU(), Linear()]
        so it becomes
        | i | layer  |
        | - | ------ |
        | 0 | Linear |
        | 1 | ReLU   |
        | 2 | Linear |

        '''
        layer_name=(f"{layer.__class__.__name__}") # layer.__class__ get object class and .__name__ gets class name as string
        parameters=layer.parameters()

        if len(parameters)==0:
            weight_shape = "-"
            bias_shape = "-"
            layer_parameter_count = 0
        else:
            weight_shape=str(parameters[0].shape) # parameters = [self.W,self.b]
            bias_shape=str(parameters[1].shape)
            layer_parameter_count = sum(parameter.size for parameter in parameters) # layer wise param count
        print(
            f"{layer_name:<20}"
            f"{weight_shape:<20}"
            f"{bias_shape:<20}"
            f"{layer_parameter_count:<15}"
        )
    print("=" * 95)

    print(
        f"{'Total Parameters':<60}"
        f"{parameter_count(model)}"
    )

    print("=" * 95)

def gradient_statistics(model:Sequential):
    for i, layer in enumerate(model.layers):
        gradients=layer.gradients()
        if len(gradients)==0:
            continue
        print(f"\nLayer {i} Gradient Statistics")
        print("-" * 40)

        for j, gradient in enumerate(gradients):
            gradient_name = ("Weights" if j == 0 else "Bias")
            print(f"\n{gradient_name} Gradient")
            print(f"Mean : {np.mean(gradient):.6f}")
            print(f"Std  : {np.std(gradient):.6f}")
            print(f"Min  : {np.min(gradient):.6f}")
            print(f"Max  : {np.max(gradient):.6f}")

def gradient_norms(model:Sequential):

    '''
    Computes the L2 norm of parameter gradients
    for each layer in the network.

    Gradient norms measure the overall magnitude
    of learning signals flowing through the model.

    Very small gradient norms:
        -> possible vanishing gradients

    Very large gradient norms:
        -> possible exploding gradients

    Useful for monitoring training stability
    and optimization behavior.
    '''

    for i,layer in enumerate(model.layers):
        gradients=layer.gradients()
        if len(gradients) == 0:
            continue
        print(f"\nLayer {i} Gradient Norms")
        print("-" * 40)

        for j,gradient in enumerate(gradients):

            gradient_name=("Weights" if j == 0 else "Bias" )
            # computes L2 norm:
            # sqrt(sum(x^2))
            norm=np.linalg.norm(gradient)
            print(
                f"{gradient_name} Gradient Norm : "
                f"{norm:.6f}"
            )