import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *

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

print("-------------------------------- Simple Regularized Logistic Regression L1 Norm ---------------------------------")
LR = LogisticRegression(learning_rate = 0.1, iterations = 100, regularization="l1")
theta = LR.fit_autograd(X, y)
y_hat = LR.predict(X)
X = np.array(X)
y = np.array(y)
print("Accuracy with L1 regularization is --> ", accuracy(y_hat, y))

dev = np.std(theta)
mn = np.mean(theta)
p = dev/2 + mn
imp_features = []
for j in range(len(theta)):
    if (theta[j]>p):
        imp_features.append(j)
print("important features after L1 regularization are --> ",  *imp_features)


print("-------------------------------- Simple Regularized Logistic Regression L2 Norm ---------------------------------")
LR = LogisticRegression(learning_rate = 0.1, iterations = 100, regularization="l2")
theta = LR.fit_autograd(X, y)
y_hat = LR.predict(X)
X = np.array(X)
y = np.array(y)
print("Accuracy with L2 regularization is --> ", accuracy(y_hat, y))


print("------------------------------- Cross Validation ----------------------------------")

def predict_cross_val(X, theta):
    bias = np.ones((X.shape[0], 1))
    X = np.append(bias, X, axis=1)
    z = 1/(1+np.exp(-(X.dot(theta))))		
    y_hat = np.where(z>0.5, 1, 0)
    return y_hat

print("------------------------------ Lambda for L1 Norm ----------------------------")

data = load_breast_cancer()
X = pd.DataFrame(data.data)
y = pd.Series(data.target)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X = np.array(X)
y = np.array(y)

X = pd.DataFrame(X)
X['y'] = y
X = np.array(X)
k_fold = KFold(3)

final_lambdas = []
final_accuracies = []

models = []
penalties = [0.1, 1.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0, 800.0]
fold = 1
for train, test in k_fold.split(X):
    print("---------------------------- FOLD - ", fold, "-----------------------------")
    val_fold = KFold(5)    
    results = {}
    for k in penalties:
        avg_acc = 0
        for train_val, test_val in val_fold.split(train):
            valid_set = X[train_val]
            test_set = X[test_val]
            valid_set = pd.DataFrame(valid_set)

            y_train = valid_set[valid_set.shape[1]-1]
            valid_set = valid_set.drop(valid_set.shape[1]-1, axis=1)
        
            test_set = pd.DataFrame(test_set)
            y_test = test_set[test_set.shape[1]-1]
            test_set = test_set.drop(test_set.shape[1]-1, axis=1)

            LR = LogisticRegression(learning_rate=0.01, iterations = 100, regularization='l1', lmbda = k)
            theta = LR.fit_autograd(valid_set, y_train)
            y_hat = LR.predict_cross_val(test_set, theta)
            avg_acc += accuracy(y_hat, y_test)
            
        avg_acc = avg_acc/5
        results[LR] = avg_acc

    X_test = X[test]
    X_test = pd.DataFrame(X_test)
    yy_test = X_test[X_test.shape[1]-1]
    X_test = X_test.drop(X_test.shape[1]-1,axis=1)        
    
    best_model = max(results, key=results.get)
    models.append(best_model)
    yhat3 = best_model.predict(X_test)
    acc = accuracy(yy_test, yhat3)
    final_accuracies.append(acc)
    fold+=1
    print("accuracy --> ", acc,  "best_l1_penalty --> ", best_model.lmbda)

print("Accuracy --> ", np.mean(final_accuracies), " with deviation of ", np.std(final_accuracies))

print("------------------------------ Lambda for L2 Norm ----------------------------")

data = load_breast_cancer()
X = pd.DataFrame(data.data)
y = pd.Series(data.target)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X = np.array(X)
y = np.array(y)

X = pd.DataFrame(X)
X['y'] = y
X = np.array(X)
k_fold = KFold(3)

final_lambdas = []
final_accuracies = []

models = []
penalties = [0.1, 1.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0, 800.0]
fold = 1
for train, test in k_fold.split(X):
    print("---------------------------- FOLD - ", fold, "-----------------------------")
    val_fold = KFold(5)    
    results = {}
    for k in penalties:
        avg_acc = 0
        for train_val, test_val in val_fold.split(train):
            valid_set = X[train_val]
            test_set = X[test_val]
            valid_set = pd.DataFrame(valid_set)

            y_train = valid_set[valid_set.shape[1]-1]
            valid_set = valid_set.drop(valid_set.shape[1]-1, axis=1)
        
            test_set = pd.DataFrame(test_set)
            y_test = test_set[test_set.shape[1]-1]
            test_set = test_set.drop(test_set.shape[1]-1, axis=1)

            LR = LogisticRegression(learning_rate=0.01, iterations = 100, regularization='l2', lmbda = k)
            theta = LR.fit_autograd(valid_set, y_train)
            y_hat = LR.predict_cross_val(test_set, theta)
            avg_acc += accuracy(y_hat, y_test)
            
        avg_acc = avg_acc/5
        results[LR] = avg_acc

    X_test = X[test]
    X_test = pd.DataFrame(X_test)
    yy_test = X_test[X_test.shape[1]-1]
    X_test = X_test.drop(X_test.shape[1]-1,axis=1)        
    
    best_model = max(results, key=results.get)
    models.append(best_model)
    yhat3 = best_model.predict(X_test)
    acc = accuracy(yy_test, yhat3)
    final_accuracies.append(acc)
    fold+=1
    print("accuracy --> ", acc,  "best_l2_penalty --> ", best_model.lmbda)

print("Accuracy --> ", np.mean(final_accuracies), " with deviation of ", np.std(final_accuracies))