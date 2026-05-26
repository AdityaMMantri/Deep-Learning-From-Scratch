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

import numpy as np
from NN.sequential import Sequential
from NN.losses import Loss
from NN.optimizers import Optimizer

class Trainer:
    def __init__(self,model:Sequential,loss_function:Loss,optimizer:Optimizer):
        self.model=model
        self.loss_function=loss_function
        self.optimizer=optimizer

    def train(self,X,y,epochs,batch_size,verbose:bool):
        number_of_samples=X.shape[0]

        if batch_size is None:
            batch_size=number_of_samples
        
        number_of_batches=int(np.ceil(number_of_samples/batch_size))
        loss_history=[]

        for epoch in range(epochs):
            indices=np.random.permutation(number_of_samples) # this is used for shuffle the data set to avoid any ordering or data bias
            X_shuffled=X[indices]
            y_shuffled=y[indices] # new target and new data set
            epoch_loss=0.0

            for starting_index in range(0,number_of_samples,batch_size): # (start, stop, step)
                ending_index=starting_index+batch_size
                #current batch
                batch_X=X_shuffled[starting_index:ending_index]
                batch_y=y_shuffled[starting_index:ending_index]

                # FORWARD PASS
                predictions=self.model.forward(inputs=batch_X)

                # LOSS
                loss=self.loss_function.forward(predictions=predictions,targets=batch_y)
                epoch_loss+=loss

                # INITIAL GRADIENT
                gradient=self.loss_function.backward()

                # BACKPROPGATION
                self.model.backward(gradient=gradient)

                # OPTIMIZER UPDATE
                self.optimizer.update_step(self.model)
            epoch_loss/=number_of_batches
            loss_history.append(epoch_loss)

            if verbose:
                print(
                    f"Epoch "
                    f"{epoch + 1}/{epochs}"
                    f" | Loss: {epoch_loss:.6f}")
        return loss_history

#Trainer is basically a high-level orchetration/controller wrapper around the ENTIRE learning pipeline.
#It treats the whole neural network as ONE black-box model object.
#Instead of managing every single layer it just gives a much more user friendly approch by giving a single model object