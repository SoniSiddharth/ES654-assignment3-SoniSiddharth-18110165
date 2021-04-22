import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.multiLogisticRegression import LogisticRegression
from metrics import *
import matplotlib.colors as cma

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay


def _one_hot(y, n_labels, dtype):
    mat = np.zeros((len(y), n_labels))
    for i, val in enumerate(y):
        mat[i, int(val)] = 1
    return mat.astype(dtype)

X = load_digits().data
y = load_digits().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
y_encoded = _one_hot(y,10,dtype="float")

print("----------------------------- Multi Class Regression ------------------------------")

LR = LogisticRegression(learning_rate = 0.01, iterations = 2000, regularization=None)
LR.fit_multiclass(X,y_encoded)
y_hat = LR.predict_multi(X)
print(accuracy(y_hat, y))

print("----------------------------- Autograd Multi Class Regression ------------------------------")

LR = LogisticRegression(learning_rate = 0.1, iterations = 20, regularization=None)
LR.fit_autograd(X,y_encoded)
y_hat = LR.predict_multi(X)
print(accuracy(y_hat, y))

def predict_multi(X, theta):
    bias = np.ones((X.shape[0], 1))
    X = np.append(bias, X, axis=1)
    Z = 1/(1+np.exp(-(X.dot(theta))))		
    Y = []
    for i in Z:
        Y.append(np.argmax(i))
    return Y

X = pd.DataFrame(X)
y = pd.Series(y)
X['y'] = y
X = np.array(X)
k_fold = KFold(4, True, 1)
avg_accuracy = 0
results = {}

for train, test in k_fold.split(X):
    valid_set = X[train]
    test_set = X[test]
    valid_set = pd.DataFrame(valid_set)
    y_train = valid_set[valid_set.shape[1]-1]
    valid_set = valid_set.drop(valid_set.shape[1]-1, axis=1)

    test_set = pd.DataFrame(test_set)
    y_test = test_set[test_set.shape[1]-1]
    test_set = test_set.drop(test_set.shape[1]-1, axis=1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train = _one_hot(y_train, 10, dtype="float")
    LR = LogisticRegression(learning_rate=0.1, iterations = 100, regularization=None)
    theta = LR.fit_multiclass(valid_set, y_train)
    y_hat = LR.predict_multi(test_set)
    accu = accuracy(y_hat, y_test)

    results[LR] = accu

best_model = max(results, key=results.get)
best_theta = best_model.coef_
X = pd.DataFrame(X)
X = X.drop(X.shape[1]-1, axis=1)
X = np.array(X)
yHat = predict_multi(X,best_theta)

mat = confusion_matrix(y, yHat)
# print(mat)

classes = np.array([0,1,2,3,4,5,6,7,8,9])
disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=classes)
disp.plot()
plt.savefig("./logisticRegression/plots/q3_ConfusionMatrix.png")
plt.show()

print("-------------------------------- PCA -------------------------------------")

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)

x_axis = X[:, 0]
y_axis = X[:, 1]
cdict = {0:'yellow', 1: 'red', 2: 'blue', 3: 'green', 4:'black', 5:'pink', 6:'purple', 7:'brown', 8:'orange', 9:'gold'}

fig, ax = plt.subplots()
for g in np.unique(y):
    ix = np.where(y == g)
    ax.scatter(x_axis[ix], y_axis[ix], c = cdict[g], label = g, s = 10)
ax.legend()
plt.savefig("./logisticRegression/plots/q3_PCA_plot.png")
plt.show()