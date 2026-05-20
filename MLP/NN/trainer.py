'''
This file controls the complete training process of the neural network.

It connects:
 - model
 - loss function
 - optimizer
 - training data

1. Run the training loop
2. Perform forward propagation
3. Compute loss
4. Start backward propagation
5. Update model parameters using optimizer
6. Repeat for multiple epochs

 ---------------------------------------------------------
 TRAINING FLOW
 ---------------------------------------------------------

 Input Data
     ↓
 Forward Pass
     ↓
 Compute Loss
     ↓
 Backward Pass
     ↓
 Compute Gradients
     ↓
 Optimizer Updates Weights
     ↓
 Repeat
'''