'''
Input
→ Forward Pass
→ Prediction
→ Loss
→ Backward Pass
→ Gradient Computation
→ Weight Update
→ Repeat 
'''
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from NN.sequential import Sequential
from NN.linear import Linear
from NN.optimizers import SGD
from NN.activations import ReLU
from NN.losses import MSELoss
from NN.trainer import Trainer

from interpretability.weights import (
    plot_weight_heatmaps,
    plot_weight_distribution
)

data=fetch_california_housing()
X=data.data
y=data.target.reshape(-1,1)
print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

model=Sequential(
    Linear(
        input_features=8,
        output_features=64,
        initialization="he"
    ),
    ReLU(),
    Linear(
        input_features=64,
        output_features=32,
        initialization="he"
    ),
    ReLU(),
    Linear(input_features=32,
           output_features=16,
           initialization="he"
    ),
    ReLU(),
    Linear(
        input_features=16,
        output_features=1,
        initialization="xe"
    )
)

loss_function=MSELoss()

optimizer=SGD(learning_rate=0.001)

trainer=Trainer(
    model=model,
    loss_function=loss_function,
    optimizer=optimizer)

loss_history = trainer.train(
    X=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    verbose=True)


# -------------------------------------------------
# TEST PREDICTIONS
# -------------------------------------------------

predictions = model.forward(X_test)

print("\nPredictions Shape:")

print(predictions.shape)

print("\nFirst 10 Predictions:\n")

print(predictions[:10])

plot_weight_heatmaps(model)

plot_weight_distribution(model)

# =================================================
# PYTORCH COMPARISON
# =================================================
# This section uses DIFFERENT variable names
# so it does not conflict with your custom NN.
# =================================================

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import (
#     TensorDataset,
#     DataLoader
# )


# # =================================================
# # NUMPY → TENSOR CONVERSION
# # =================================================

# torch_train_inputs = torch.tensor(
#     X_train,
#     dtype=torch.float32
# )

# torch_train_targets = torch.tensor(
#     y_train,
#     dtype=torch.float32
# )

# torch_test_inputs = torch.tensor(
#     X_test,
#     dtype=torch.float32
# )

# torch_test_targets = torch.tensor(
#     y_test,
#     dtype=torch.float32
# )


# # =================================================
# # DATALOADER
# # =================================================

# torch_dataset = TensorDataset(
#     torch_train_inputs,
#     torch_train_targets
# )

# torch_loader = DataLoader(
#     torch_dataset,
#     batch_size=32,
#     shuffle=True
# )


# # =================================================
# # BUILD PYTORCH MODEL
# # =================================================

# torch_network = nn.Sequential(

#     nn.Linear(8,64),

#     nn.ReLU(),

#     nn.Linear(64,32),

#     nn.ReLU(),

#     nn.Linear(32,16),

#     nn.ReLU(),

#     nn.Linear(16,1)
# )


# torch_loss_function = nn.MSELoss()

# torch_optimizer = optim.SGD(
#     torch_network.parameters(),
#     lr=0.001)

# torch_epochs = 100

# for epoch in range(torch_epochs):

#     accumulated_loss = 0.0

#     for current_inputs, current_targets in torch_loader:

#         # clear previous gradients
#         torch_optimizer.zero_grad()

#         # forward pass
#         torch_predictions = torch_network(
#             current_inputs
#         )

#         # compute loss
#         torch_loss = torch_loss_function(
#             torch_predictions,
#             current_targets
#         )

#         # backward pass
#         torch_loss.backward()

#         # update weights
#         torch_optimizer.step()

#         accumulated_loss += (
#             torch_loss.item()
#         )

#     mean_loss = (
#         accumulated_loss
#         /
#         len(torch_loader)
#     )

#     print(
#         f"[PyTorch] "
#         f"Epoch {epoch+1}/{torch_epochs}"
#         f" | Loss: {mean_loss:.6f}"
#     )


# # =================================================
# # TEST PREDICTIONS
# # =================================================

# with torch.no_grad():

#     torch_test_predictions = (
#         torch_network(torch_test_inputs)
#     )

# print("\n[PyTorch] Prediction Shape:\n")

# print(torch_test_predictions.shape)

# print("\n[PyTorch] First 10 Predictions:\n")

# print(torch_test_predictions[:10])


# # =================================================
# # COMPARISON
# # =================================================

# print("\n=================================================")
# print("CUSTOM NN vs PYTORCH")
# print("=================================================")

# print("\nCustom NN Prediction Shape:")

# print(predictions.shape)

# print("\nPyTorch Prediction Shape:")

# print(torch_test_predictions.shape)

# print("\nCustom NN First Prediction:")

# print(predictions[0])

# print("\nPyTorch First Prediction:")

# print(
#     torch_test_predictions[0]
#     .numpy()
# )