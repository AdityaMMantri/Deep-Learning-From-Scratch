'''
This file implements the Dense/Linear layer of the neural network.
This is the MAIN computational layer where learning actually happens.
The Linear layer performs:

    output = input @ weights + bias

1. Store trainable parameters
    - weights
    - biases

2. Perform forward propagation
    - matrix multiplication
    - bias addition

3. Store inputs during forward pass because backward propagation needs them later

4. Perform backward propagation
    - compute weight gradients
    - compute bias gradients
    - compute input gradients

 5. Send gradients backward to previous layer

FORWARD PASS

Input Shape:
    (batch_size, input_features)

Weight Shape:
    (input_features, output_features)

 Output Shape:
    (batch_size, output_features)

BACKWARD PASS

During backpropagation this layer computes:

1. Gradient wrt weights
    -> tells how weights should change

2. Gradient wrt bias
    -> tells how biases should change

3. Gradient wrt input
    -> passed backward to previous layer
''' 

import numpy as np
from module import Module

class Linear(Module):
    def __init__(self,input_features,output_features,initialization):
        self.in_features=input_features
        self.out_features=output_features

        if initialization=="he":
            self.W=np.random.randn(input_features,output_features)*np.sqrt(2/input_features)
        elif initialization=="xavier":
            self.W=np.random.randn(input_features,output_features)*np.sqrt(2/(input_features+output_features))
        else:
            raise ValueError("initialization should be he or xavier")
        self.b=np.zeros((1,output_features))

        self.gradient_W=np.zeros_like(self.W)
        self.gradient_b=np.zeros_like(self.b)        
    
    def forward(self,inputs):
        self.inputs=inputs # caching inputs for backprop
        Z=np.dot(inputs,self.W) + self.b # weighted sum on inputs + bias
        self.output=Z # caching output aslo 
        return Z
    
    def backward(self,gradient_outputs):
        # recieve the gradient output from the l+1 layer
        # then compute delta L/ delta W
        # then compute delta L/ delta b
        # then compute the gradient output for l-1 layer
        batch_size=self.inputs.shape[0]
        self.gradient_W=np.dot(self.inputs.T,gradient_outputs)/batch_size
        self.gradient_b=np.sum(gradient_outputs,axis=0,keepdims=True)/batch_size
        gradient_inputs=np.dot(gradient_outputs,self.W.T)
        return gradient_inputs
    
    def parameters(self):
        return [self.W,self.b]
    
    def gradients(self):
        return [self.gradient_W,self.gradient_b]
    
# these are the gradients only for the linear layer excluding the activation and other only the Wx+b

    # recieve gradient from next layer
    #
    # gradient_outputs = dL/dZ
    #
    # tells how sensitive loss is
    # wrt current layer outputs
    # -----------------------------------
    # Weight Gradient
    #
    # dL/dW = X^T . gradient_outputs
    #
    # derived using chain rule:
    #
    # dL/dW = dL/dZ * dZ/dW
    #
    # since:
    # Z = XW + b
    #
    # derivative wrt W becomes X
    # -----------------------------------
    # Bias Gradient
    #
    # dL/db = sum(gradient_outputs)
    #
    # because bias is added equally
    # to every sample in batch
    # -----------------------------------
    # Input Gradient
    #
    # dL/dX = gradient_outputs . W^T
    #
    # derived using chain rule:
    #
    # dL/dX = dL/dZ * dZ/dX
    #
    # derivative wrt X becomes W
    #
    # this gradient gets passed
    # to previous layer
    # -----------------------------------