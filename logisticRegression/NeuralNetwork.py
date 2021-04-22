import numpy as np 
from autograd import numpy as np, elementwise_grad

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    x[x<0]=0
    return x

def identity(x):
    return x

def mse(y_pred, y_true):
    return np.mean(np.power(y_true-y_pred, 2))

class Layer():
    def __init__(self, in_nodes, out_nodes, activation, func):
        self.weights = np.random.rand(in_nodes, out_nodes) - 0.5
        self.bias = np.random.rand(1, out_nodes) - 0.5
        self.activation = activation
        self.func = func

    def forwardprop(self, X):
        self.X = X
        if self.activation:
            self.z = self.func(self.X)
        else:
            self.z = np.dot(self.X, self.weights) + self.bias
        return self.z

    def backwardprop(self, dJdZ, lr):
        if self.activation:
            agrad = elementwise_grad(sigmoid)
            error_x = agrad(self.X)*dJdZ       # calculating dJ/dX by dJ/dZ * dZ/dX, downstream gradient = local gradient * upstream gradient 
        else:
            error_x = np.dot(dJdZ, self.weights.T)   # dZ/dX = W --> dJ/dX = dJ/dZ * W     
            error_w = np.dot(self.X.T, dJdZ)        # similarly dW = X * dJ/dZ
            self.weights -= lr * error_w 
            self.bias -= lr * dJdZ
        return error_x

class NeuralNetwork():
    def __init__(self, learning_rate=0.1, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.layerslist = []

    def add(self, layer):
        self.layerslist.append(layer)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        samples = X.shape[0]
        for i in range(self.iterations):
            err = 0
            for j in range(samples):
                current = X[j]
                for layer in self.layerslist:
                    current = layer.forwardprop(current)
                err += mse(current, y[j])
                # backward propagation
                agrad = elementwise_grad(mse)
                error = agrad(current, y[j])      # dJ/dZ (Z = current)
                for layer in reversed(self.layerslist):
                    error = layer.backwardprop(error, self.learning_rate)
            err = err/samples
            print('epoch %d/%d   error=%f' % (i+1, self.iterations, err))

    def predict(self, X):
        samples = X.shape[0]
        y_pred = []
        for i in range(samples):
            current = X[i]
            for layer in self.layerslist:
                current = layer.forwardprop(current)
            y_pred.append(current)
        return y_pred
