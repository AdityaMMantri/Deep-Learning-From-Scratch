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