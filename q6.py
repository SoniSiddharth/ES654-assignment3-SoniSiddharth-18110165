import numpy as np
import pandas as pd
from logisticRegression.NeuralNetwork import NeuralNetwork, Layer, relu, sigmoid, mse, identity
from sklearn.datasets import load_breast_cancer, load_digits
from keras.utils import np_utils
from metrics import *
import math

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import KFold

X = load_boston().data
y = load_boston().target

scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

# ------------------------------------------- Without Validation Regression ------------------------------
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


# ------------------------------------------ 3 Fold Cross Validation ---------------------------------

print("----------------------------------- Corss Validation -------------------------------")

X = load_boston().data
y = load_boston().target

scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

X = pd.DataFrame(X)
y = pd.Series(y)

X['y'] = y
X = np.array(X)
k_fold = KFold(3, True, 1)
avg_accuracy = 0
fold = 1
for train, test in k_fold.split(X):
    valid_set = X[train]
    test_set = X[test]

    valid_set = pd.DataFrame(valid_set)
    y_train = valid_set[valid_set.shape[1]-1]
    valid_set = valid_set.drop(valid_set.shape[1]-1, axis=1)

    y_train = np_utils.to_categorical(y_train)

    test_set = pd.DataFrame(test_set)
    y_test = test_set[test_set.shape[1]-1]
    test_set = test_set.drop(test_set.shape[1]-1, axis=1)

    valid_set = np.array(valid_set)
    valid_set = valid_set.reshape(valid_set.shape[0], 1, 13)
    valid_set = valid_set.astype('float32')

    test_set = np.array(test_set)
    test_set = test_set.reshape(test_set.shape[0], 1, 13)
    test_set = test_set.astype('float32')

    NN = NeuralNetwork(iterations=100, learning_rate=0.01)

    NN.add(Layer(13, 13, activation=None, func=sigmoid))
    NN.add(Layer(13, 13, activation=True, func=sigmoid))
    NN.add(Layer(13, 13, activation=None, func=relu))
    NN.add(Layer(13, 13, activation=True, func=relu))
    NN.add(Layer(13, 1, activation=None, func=identity))
    NN.fit(valid_set, y_train)

    y_pred = NN.predict(test_set)

    acc = rmse(y_pred, y_test)
    print("RMSE of fold ", fold, " is --> ", acc[0][0])
    avg_accuracy += acc[0][0]
    fold+=1

avg_accuracy = avg_accuracy/3
print("Average RMSE on 3 Fold Testing is --> ", avg_accuracy)