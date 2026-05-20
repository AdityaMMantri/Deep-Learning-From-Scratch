'''
This file defines the Loss Functions of the neural network.

1. Compare predictions with actual targets
2. Compute a single scalar loss value
3. Compute the initial gradient for backpropagation

FORWARD PASS

Takes:
 - predicted values
 - actual values

Returns:
 - loss value

BACKWARD PASS

Computes:

    dLoss / dPrediction

This gradient starts the backward propagation process.
''' 

import numpy as np

class MSELoss:
    def forward(self,predictions,targets):
        self.predictions=predictions
        self.targets=targets # this is caching ie... we are storing values for backward pass

        loss=np.mean((targets-predictions)**2)
        return loss
    
    def backward(self):
        batch_size=self.targets.shape[0]
        gradient = (2*(self.predictions-self.targets))/batch_size
        return gradient

class BCELoss:
    def forward(self,predictions,targets):
        predictions=np.clip(predictions,1e-7,1-1e-7)

        self.predictions=predictions
        self.targets=targets

        loss = -np.mean((targets*np.log(predictions))+
                        (1-targets)*np.log(1-predictions))
        return loss
    
    def backward(self):
        batch_size=self.targets.shape[0]

        gradient = (self.predictions-self.targets)/(self.predictions*(1-self.predictions))/batch_size
        return gradient
