import numpy as np
import pandas as pd
from logisticRegression.NeuralNetwork import NeuralNetwork, Layer, relu, sigmoid, mse, identity
from sklearn.datasets import load_breast_cancer, load_digits
from keras.utils import np_utils
from metrics import *
import math

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler    

X = load_boston().data
y = load_boston().target

scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
X = X.reshape(X.shape[0], 1, 13)
X = X.astype('float32')

# Neural Network 
NN = NeuralNetwork(iterations=100, learning_rate=0.01)

NN.add(Layer(13, 13, activation=None, func=sigmoid))
NN.add(Layer(13, 13, activation=True, func=sigmoid))
NN.add(Layer(13, 13, activation=None, func=relu))
NN.add(Layer(13, 13, activation=True, func=relu))
NN.add(Layer(13, 1, activation=None, func=identity))
NN.fit(X, y)

y_pred = NN.predict(X)
print("RMSE on Boston --> ", rmse(y_pred, y)[0][0])
