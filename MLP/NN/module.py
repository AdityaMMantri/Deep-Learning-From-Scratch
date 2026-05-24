'''
This file defines the BASE blueprint for every component in the neural network.
Every layer (Linear, ReLU, Sigmoid, etc.) will inherit from this.
It ensures that all layers follow the same structure and rules.

The main purpose is to standardize:

1. forward()
   -> how the layer computes output

2. backward()
   -> how gradients flow backward through the layer

3. parameters()
   -> returns trainable weights/biases

4. gradients()
   -> returns gradients of weights/biases

Because every layer follows this common interface,
the Sequential model can dynamically loop through layers
during forward and backward propagation.
'''

class Module:
   def forward(self,inputs):
      raise NotImplementedError("Forward pass method not implemented") # every child class inherithing this class must have forward function
   
   def backward(self,gradient):
      raise NotImplementedError("Backward pass method not implemented")# every child class inherithing this class must have backward function
   
   def parameters(self):
      # every child class inheriting this class can return
      # its trainable parameters like weights and biases
      return []
   
   def gradients(self):
      # every child class inheriting this class can return
      # gradients of its trainable parameters
      return []
