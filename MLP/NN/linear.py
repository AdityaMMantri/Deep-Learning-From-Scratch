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
