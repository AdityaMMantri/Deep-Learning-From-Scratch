"""
This module contains utilities for analyzing and visualizing
neuron activations inside the neural network.

Activations represent the internal feature representations
learned by the model after each layer transformation.

By studying activations, we can understand:
- how data flows through the network
- which neurons activate for different inputs
- sparsity and dead neurons
- how representations evolve layer-by-layer

This file may include:
- activation histograms
- sparsity analysis
- dead neuron detection
- activation statistics
- layer-wise activation visualization
"""
import numpy as np
import matplotlib.pyplot as plt

from NN.sequential import Sequential
from NN.activations import ReLU

def activation_distribution(model:Sequential):
    # how neurons outputs are distributed
    # histogram of neuron outputs
    for i, layer in enumerate(model.layers):
        if hasattr(layer,"output"):
            activations=layer.output.flatten()
            plt.figure(figsize=(10,5),dpi=120)
            plt.hist(activations,bins=40,color="mediumseagreen",edgecolor="black",alpha=0.85)
            plt.title(f"Layer {i} Activation Distribution",fontsize=18,fontweight="bold")
            plt.xlabel("Activation Value",fontsize=13)
            plt.ylabel("Frequency",fontsize=13)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()

def activation_statistics(model:Sequential):
    # Activation Statistics
    for i, layer in enumerate(model.layers):
        if hasattr(layer,"output"):
            activations=layer.output
            print(f"\nLayer {i} Activation Statistics")

            print("-" * 30)

            print(f"Shape : {activations.shape}")
            print(f"Mean  : {np.mean(activations):.4f}")
            print(f"Std   : {np.std(activations):.4f}")
            print(f"Min   : {np.min(activations):.4f}")
            print(f"Max   : {np.max(activations):.4f}")

def activation_sparsity(model:Sequential,threshold=1e-3):
    for i,layer in enumerate(model.layers):
        if hasattr(layer,"output"):
            activations=layer.output

            # for Relu
            if isinstance(layer,ReLU):
                inactive=np.sum(activations==0)
            else:
                inactive=np.sum(np.abs(activations)<threshold)
            total_activations=activations.size
            sparsity=inactive/total_activations
            print(f"Layer {i} Sparsity : {sparsity:.2%}")

def activation_heatmaps(model:Sequential):
    for i,layer in enumerate(model.layers):
        if hasattr(layer,"output"):
            activations=layer.output
            plt.figure(figsize=(12,6),dpi=120)
            image=plt.imshow(activations,aspect="auto",cmap="viridis",interpolation="nearest")
            cbar=plt.colorbar(image)
            cbar.set_label("Activation Magnitude",fontsize=12)
            plt.title(f"Layer {i} Activation Heatmap",fontsize=18,fontweight="bold",pad=15)
            plt.xlabel("Neuron Index",fontsize=13)
            plt.ylabel("Sample Index",fontsize=13)
            plt.tight_layout()
            plt.show()

def activation_correlation(model:Sequential):
    '''
    computes correlation between neurons based on their activations
    high positive correlation (+1)
    -> neurons behave similarly
    -> possible redundant feature detectors
    near zero correlation (0)
    -> neurons behave independently
    high negative correlation (-1)
    -> neurons activate oppositely
    '''
    for i,layer in enumerate(model.layers):
        if hasattr(layer,"output"):
            activations=layer.output
            corr_matrix=np.corrcoef(activations,rowvar=False)# activations shape is (samples, neurons),rowvar=False tells numpy to treat
                                                             # columns (neurons) as variables so correlation is computed between neurons and not samples
            plt.figure(figsize=(8,6),dpi=120)
            image=plt.imshow(corr_matrix,cmap="coolwarm",vmin=-1,vmax=1,interpolation="nearest")# vmin and vmax define the minimum and maximum values mapped to the colormap
            cbar=plt.colorbar(image)
            cbar.set_label("Correlation",fontsize=12)
            plt.title(f"Layer {i} Activation Correlation",fontsize=18,fontweight="bold",pad=15)
            plt.xlabel("Neuron Index",fontsize=13)
            plt.ylabel("Neuron Index",fontsize=13)
            plt.tight_layout()
            plt.show()

def top_activating_neurons(model:Sequential,top_k=10):
    '''
    Identifies neurons with the highest average activation
    across the current dataset/batch.

    The activation matrix has shape:
        (samples, neurons)

    For each neuron(column), the function computes:
    mean activation across all samples

    High mean activation:
    -> neuron responds strongly/frequently
    -> possible dominant feature detector
    Low mean activation:
    -> weakly active neuron
    -> possibly less useful or specialized.
    This provides neuron-level behaviour analysis
    by showing which neurons are most active
    for the current input data.
    '''
    for i,layer in enumerate(model.layers):
        if hasattr(layer,"output"):
            activation=layer.output
            mean_activations=np.mean(activation,axis=0) # column wise again as columns represent neurons
            top_neurons=np.argsort(mean_activations)[:-top_k]
            print(f"\nLayer {i} Top Activating Neurons")
            print("-" * 40)
            for idx in reversed(top_neurons): # reversed because argsort gives ascending order
                print(
                    f"Neuron {idx} "
                    f"-> Mean Activation : "
                    f"{mean_activations[idx]:.4f}"
                )
            plt.figure(figsize=(10,5), dpi=120)
            plt.bar(np.arange(len(mean_activations)),mean_activations,color="cornflowerblue")
            plt.title(
                f"Layer {i} Mean Neuron Activations",
                fontsize=18,
                fontweight="bold"
            )
            plt.xlabel("Neuron Index",fontsize=13)
            plt.ylabel("Mean Activation",fontsize=13)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()

def activation_variance(model:Sequential):
    '''
        Measures how much each neuron activation varies
    across different input samples.

    The activation matrix has shape:
        (samples, neurons)

    For each neuron(column), the function computes:
        variance across all samples

    High variance:
        -> neuron responds differently to inputs,dynamic feature detectors
        -> possible specialized feature detector

    Low variance:
        -> neuron outputs similar values always
        -> possibly weak or redundant neuron

    This helps identify neurons that react strongly
    to specific patterns in the data.
    '''
    for i,layer in enumerate(model.layers):
        if hasattr(layer,"output"):
            activation=layer.output
            variances_activatons=np.var(variances_activatons,axis=0)
            plt.figure(figsize=(10,5),dpi=120)
            plt.bar(np.arange(len(variances_activatons)),variances_activatons,color="darkorange")
            plt.title(
                f"Layer {i} Activation Variance",
                fontsize=18,
                fontweight="bold"
            )
            plt.xlabel("Neuron Index",fontsize=13)
            plt.ylabel("Variance",fontsize=13)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()