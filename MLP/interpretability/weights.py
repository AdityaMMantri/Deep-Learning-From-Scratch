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

import numpy as np
import matplotlib.pyplot as plt

from NN.linear import Linear
from NN.sequential import Sequential

def plot_weight_heatmaps(model:Sequential):
    # weight heatmap
    for i,layer in enumerate(model.layers):
        if isinstance(layer,Linear):
            plt.figure(figsize=(14,6),dpi=120)
            image=plt.imshow(layer.W,aspect="equal",cmap="magma",interpolation="nearest")
            cbar=plt.colorbar(image)
            cbar.set_label("Weight Magnitude",fontsize=12)
            plt.title(f"Layer {i} Weight Matrix",fontsize=18,fontweight="bold",pad=15)
            plt.xlabel("Neurons",fontsize=14)
            plt.ylabel("Input Features",fontsize=14)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(False)
            plt.tight_layout()
            plt.show()
# Go through every layer in the network.If the current layer is Linear,then visualize its weights.
# Ignore activation layers.

def plot_weight_distribution(model:Sequential):
    # histogram of weights/distribution
    for i,layer in enumerate(model.layers):
        if isinstance(layer,Linear):
            plt.figure(figsize=(12,6))
            plt.hist(layer.W.flatten(),bins=40,color="royalblue",edgecolor="black",alpha=0.85)
            plt.title(f"Layer {i} Weight Distribution",fontsize=18,fontweight="bold",pad=15)
            plt.xlabel("Weight Values",fontsize=14)
            plt.ylabel("Frequency",fontsize=14)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.grid(alpha=0.25,linestyle="--")
            plt.tight_layout()
            plt.show()

def weight_statistics(model:Sequential):
    for i, layer in enumerate(model.layers):
        if isinstance(layer,Linear):
            weights=layer.W
            print(f"\nLayer {i} Statistics")

            print("-" * 30)

            print(f"Shape : {weights.shape}")
            print(f"Mean  : {np.mean(weights):.4f}")
            print(f"Std   : {np.std(weights):.4f}")
            print(f"Min   : {np.min(weights):.4f}")
            print(f"Max   : {np.max(weights):.4f}")

def neuron_magnitude(model:Sequential):
    # to calculate the magnitude of each neurons weight vector
    for i,layer in enumerate(model.layers):
        if isinstance(layer,Linear):
            magnitude=np.linalg.norm(layer.W,axis=0)
            plt.figure(figsize=(10,5))
            plt.bar(np.arange(len(magnitude)),magnitude,color="darkorange")
            plt.title(f"Layer {i} Neuron Norms",fontsize=18,fontweight="bold")
            plt.xlabel("Neuron Index")
            plt.ylabel("L2 Norm")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()