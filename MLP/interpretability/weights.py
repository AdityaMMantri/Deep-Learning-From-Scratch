"""
This module contains utilities for analyzing and visualizing
the learned weight matrices of the neural network.

The weights in a layer define the linear transformation:

    y = Wx + b

By studying these matrices, we can understand:
- what features neurons learn
- how the network transforms data
- neuron importance and redundancy
- hidden geometric structure inside the model

This file may include:
- weight heatmaps
- weight histograms
- neuron norm analysis
- similarity analysis
- matrix rank / SVD analysis
"""