import numpy as np
import pandas as pd
from logisticRegression.NeuralNetwork import NeuralNetwork, Layer, relu, sigmoid, mse, identity
from sklearn.datasets import load_digits
from keras.datasets import mnist
from keras.utils import np_utils
from metrics import *
from sklearn.model_selection import KFold

X = load_digits().data
y = load_digits().target

# ------------------------- without Cross validation ---------------------------------

X = X.reshape(X.shape[0], 1, 64)
X = X.astype('float32')
y_encoded = np_utils.to_categorical(y)

# Network
NN = NeuralNetwork(iterations=20, learning_rate=0.5)
NN.add(Layer(64, 32, activation=None, func=sigmoid))
NN.add(Layer(64, 32, activation=True, func=sigmoid))
NN.add(Layer(32, 16, activation=None, func=relu))
NN.add(Layer(32, 16, activation=True, func=relu))
NN.add(Layer(16, 10, activation=None, func=identity))
NN.add(Layer(16, 10, activation=True, func=identity))
NN.fit(X, y_encoded)
y_pred = NN.predict(X)

y_hat = []
for i in y_pred:
    y_hat.append(np.argmax(i))

acc = accuracy(y_hat, y)
print("Accuracy is --> ", acc)


# --------------------------------------- Cross validation ----------------------------------

print("------------------------------- Cross Validation --------------------------------------")

X = load_digits().data
y = load_digits().target

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
    valid_set = valid_set.reshape(valid_set.shape[0], 1, 64)
    valid_set = valid_set.astype('float32')

    test_set = np.array(test_set)
    test_set = test_set.reshape(test_set.shape[0], 1, 64)
    test_set = test_set.astype('float32')

    NN = NeuralNetwork(iterations=20, learning_rate=0.5)
    NN.add(Layer(64, 32, activation=None, func=sigmoid))
    NN.add(Layer(64, 32, activation=True, func=sigmoid))
    NN.add(Layer(32, 16, activation=None, func=relu))
    NN.add(Layer(32, 16, activation=True, func=relu))
    NN.add(Layer(16, 10, activation=None, func=identity))
    NN.add(Layer(16, 10, activation=True, func=identity))
    NN.fit(valid_set, y_train)
    y_pred = NN.predict(test_set)

    y_hat = []
    for i in y_pred:
        y_hat.append(np.argmax(i))

    y_test = np.array(y_test)
    acc = accuracy(y_hat, y_test)
    print("Accuracy of fold ", fold, " is --> ", acc)
    avg_accuracy += acc
    fold+=1

avg_accuracy = avg_accuracy/3
print("Average accuracy on 3 Fold Testing is --> ", avg_accuracy*100, " %")