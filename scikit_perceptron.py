import numpy as np
from sklearn.linear_model import Perceptron

class SKPerceptron:
    def __init__(self,epochs,learning_rate):
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.weights=None
        self.bias=None
        self.losses=[]
    
    def forward_pass(self,X):
        return np.dot(X,self.weights) + self.bias

    def activation_function(self,z):
        return np.where(z>0,1,-1)
    
    def perceptron_loss(self,margin):
        return max(0,-margin)

    def fit(self,X,y):
        n_sample,n_feature=X.shape
        self.weights=np.zeros(n_feature)
        self.bias=0.0
        for epoch in range(self.epochs):
            epoch_loss=0.0
            for i in range(len(X)):
                xi=X[i]
                yi=y[i]
                z=self.forward_pass(X=xi)
                margin=yi*z
                loss=self.perceptron_loss(margin)
                epoch_loss+=loss
                if margin <= 0:
                    self.weights+=(self.learning_rate*yi*xi) # w← w + learning_rate*yi​*xi
                    self.bias+=(self.learning_rate*yi) # b← b + learning_rate*yi​

            epoch_loss = epoch_loss / n_sample
            self.losses.append(epoch_loss)

    def predict(self,X):
        z = self.forward_pass(X)
        return self.activation_function(z)

X = np.array([
    [4.2, 1.3],
    [3.8, 2.7],
    [5.1, 2.2],
    [4.7, 3.9],
    [6.0, 1.8],
    [5.5, 4.1],
    [3.5, 3.2],

    [1.2, -0.3],
    [0.7, 0.4],
    [1.5, 0.2],

    [-3.5, -4.0],
    [-4.2, -2.8],
    [-5.1, -3.7],
    [-4.8, -5.2],
    [-6.0, -2.5],
    [-5.7, -4.1],
    [-3.9, -5.5],
    [-0.5, 1.0],
    [-1.2, 0.3],
    [-0.8, -0.2]])

y = np.array([
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1,

    -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1
])
custom_model = SKPerceptron(epochs=20,learning_rate=1.0)
custom_model.fit(X, y)
custom_predictions = custom_model.predict(X)
custom_accuracy = np.mean(custom_predictions == y)

sk_model = Perceptron(
    max_iter=20,
    eta0=1.0,
    tol=None,
    shuffle=False)

sk_model.fit(X, y)
sk_predictions = sk_model.predict(X)
sk_accuracy = np.mean(sk_predictions == y)

print("\n================ CUSTOM PERCEPTRON ================\n")

print("Predictions:")
print(custom_predictions)

print("\nAccuracy:")
print(custom_accuracy)

print("\nWeights:")
print(custom_model.weights)

print("\nBias:")
print(custom_model.bias)

print("\nLosses Per Epoch:")
print(custom_model.losses)


print("\n================ SKLEARN PERCEPTRON ================\n")

print("Predictions:")
print(sk_predictions)

print("\nAccuracy:")
print(sk_accuracy)

print("\nWeights:")
print(sk_model.coef_)

print("\nBias:")
print(sk_model.intercept_)