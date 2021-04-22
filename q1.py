import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
import matplotlib.colors as cma

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

data = load_breast_cancer()
X = pd.DataFrame(data.data)
y = pd.Series(data.target)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X = np.array(X)
y = np.array(y)

print("-------------------------------- Simple Unregularized Logistic Regression ---------------------------------")
LR = LogisticRegression(learning_rate = 0.1, iterations = 1000)
LR.fit(X, y)
y_hat = LR.predict(X)
X = np.array(X)
y = np.array(y)
# print(y_hat)
print("accuracy of normal Logistic Regression --> ", accuracy(y_hat, y))


print("-------------------------------- Autograd Unregularized Logistic Regression ---------------------------------")

LR = LogisticRegression(learning_rate = 0.1, iterations = 1000)
LR.fit_autograd(X, y)
y_hat = LR.predict(X)
X = np.array(X)
y = np.array(y)
# print(y_hat)
print("accuracy of normal Logistic Regression --> ", accuracy(y_hat, y))

print("------------------------------ K Fold Validation -------------------------")


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
    # valid_set, mu, sigma = scale_features(valid_set)

    test_set = pd.DataFrame(test_set)
    y_test = test_set[test_set.shape[1]-1]
    test_set = test_set.drop(test_set.shape[1]-1, axis=1)
    # test_set = transform_features(test_set, mu, sigma)

    LR = LogisticRegression(learning_rate = 0.1, iterations = 1000)
    LR.fit(valid_set, y_train)
    y_hat = LR.predict(test_set)
    acc = accuracy(y_hat, y_test)
    print("Accuracy of fold ", fold, " is --> ", acc)
    avg_accuracy += acc
    fold+=1

avg_accuracy = avg_accuracy/3
print("Average accuracy on 3 Fold Testing is --> ", avg_accuracy*100)


LR = LogisticRegression(learning_rate = 0.1, iterations = 1000)
X = np.array(X)
y = np.array(y)
X_featured = X[:, [1,2]]
theta = LR.fit(X_featured, y)
LR.plot_decision_boundary(X_featured, y, theta)