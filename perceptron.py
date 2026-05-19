import numpy as np
from sklearn.linear_model import Perceptron as SkPerceptron

np.random.seed(42)

class Perceptron:
    def __init__(self,n_features,learning_rate,iterations):
        self.weights=np.random.randn(n_features)
        self.bias=np.random.randn()
        self.lr=learning_rate
        self.iterations=iterations

    def forward_pass(self,X):
        return np.dot(self.weights,X) + self.bias

    def activation_function(self,z):
        if z>=0:
            return 1
        else:
            return 0

    def predict(self,X):
        z=self.forward_pass(X)
        return self.activation_function(z)

    def fit(self,X,y):
        for epoch in range(self.iterations):
            total_errors=0
            for i in range(len(X)):
                xi=X[i] # classical sample wise perceptron learning algorithm
                yi=y[i] # current target
                y_pred=self.predict(X=xi)
                error=yi-y_pred
                self.weights+=self.lr*error*xi # update rule is w=w+leanring_rate*(ytrue​−ypred​*)x
                self.bias+=self.lr*error # b=b+learning_rate*(ytrue​−ypred​)

                if error!=0:
                    total_errors+=1
            print(f"Epoch {epoch+1}, Errors: {total_errors}")

# BINARY INPUT AND GATE
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

y=np.array([0,0,0,1])

model=Perceptron(
    n_features=2,
    learning_rate=0.1,
    iterations=15)

model.fit(X, y)

sk_model=SkPerceptron(
    eta0=0.1,
    max_iter=10,
    tol=None,
    shuffle=False)

sk_model.fit(X,y)

print("\nMy Model Predictions")

for sample in X:

    pred = model.predict(sample)

    print(sample, "->", pred)


print("\nSklearn Predictions")

for sample in X:

    pred = sk_model.predict([sample])[0]

    print(sample, "->", pred)

print("\nMy Weights:")
print(model.weights)

print("My Bias:")
print(model.bias)


print("\nSklearn Weights:")
print(sk_model.coef_)

print("Sklearn Bias:")
print(sk_model.intercept_)