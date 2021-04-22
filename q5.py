import numpy as np
import pandas as pd
from logisticRegression.NeuralNetwork import NeuralNetwork, Layer, relu, sigmoid, mse, identity
from sklearn.datasets import load_digits
from keras.datasets import mnist
from keras.utils import np_utils
from metrics import *

X = load_digits().data
y = load_digits().target

X = X.reshape(X.shape[0], 1, 64)
X = X.astype('float32')

y = np_utils.to_categorical(y)

# Network
NN = NeuralNetwork(iterations=20, learning_rate=0.5)
NN.add(Layer(64, 32, activation=None, func=sigmoid))
NN.add(Layer(64, 32, activation=True, func=sigmoid))
NN.add(Layer(32, 16, activation=None, func=relu))
NN.add(Layer(32, 16, activation=True, func=relu))
NN.add(Layer(16, 10, activation=None, func=identity))
NN.add(Layer(16, 10, activation=True, func=identity))
NN.fit(X, y)
y_pred = NN.predict(X)

y_hat = []
for i in y_pred:
    y_hat.append(np.argmax(i))
y_pred = np.argmax(y_pred ,axis=1)

y_original = []
for i in y:
    y_original.append(np.argmax(i))

print("Accuracy on load digits is --> ", accuracy(y_hat, y_original))
