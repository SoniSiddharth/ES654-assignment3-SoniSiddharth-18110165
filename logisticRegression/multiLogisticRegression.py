# Importing libraries
import numpy as np
import pandas as pd
import warnings
import autograd.numpy as np
from autograd import elementwise_grad
import autograd as grad
from scipy.special import expit, logit
warnings.filterwarnings( "ignore" )

def cost_function(theta,X,y, lmbda):
	y_hat = np.dot(X,theta)
	z = 1/(1 + np.exp(-y_hat))
	return -np.sum(y*(np.log(z)) + (1-y)*np.log(1 - z))
    
def cost_function_l1(theta,X,y, lmbda):
	y_hat = np.dot(X,theta)
	z = 1/(1 + np.exp(-y_hat))
	return -np.sum(y*(np.log(z)) + (1-y)*np.log(1 - z)) + lmbda*(abs(theta))

def cost_function_l2(theta,X,y, lmbda):
	y_hat = np.dot(X,theta)
	z = 1/(1 + np.exp(-y_hat))
	return -np.sum(y*(np.log(z)) + (1-y)*np.log(1 - z)) + lmbda*(theta.T*theta)

class LogisticRegression() :
	def __init__(self, learning_rate=0.001, iterations=1000, lmbda=0.1, regularization=None, fit_intercept=True):
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.lmbda = lmbda
		self.regularization = regularization
		self.fit_intercept = fit_intercept

	def multi_cost(self, theta, X, y):
		cost= 0
		N = X.shape[0]
		for i in range(N):
			k = np.argmax(y[i])
			z = np.dot(X[i],theta)
			denominator = 0
			denominator = np.sum(np.exp(z))
			numerator = np.exp(np.dot(X[i],theta[:,k]))
			probab = np.log(numerator/denominator)
			cost += probab
		cost = -cost
		return cost

	def fit_multiclass(self,X,y):
		self.X = np.array(X)
		self.y = np.array(y)
		self.labels = np.unique(y)

		if self.fit_intercept==True:
			bias = np.ones((self.X.shape[0], 1))
			self.X = np.append(bias, self.X, axis=1)
		
		self.features = self.X.shape[1]
		self.samples = len(self.X)
		self.coef_= np.ones((self.features,y.shape[1]))

		for i in range(self.iterations):
			sm = 0
			err = 1/(1 + np.exp(-(self.X.dot(self.coef_))))-y
			for j in range(self.y.shape[1]):
				sm += np.exp(self.X.dot(self.coef_[:,j]))
			for j in range(self.y.shape[1]):
				z = np.exp(self.X.dot(self.coef_[:,j]))/sm   #1797x65x1797x1
				err = -(self.y[:,j] -z)
				self.coef_[:,j]= self.coef_[:,j] - self.learning_rate * err.dot(self.X)/self.samples
		return self.coef_

	def fit_autograd(self, X, y):
		self.X = np.array(X)
		self.y = np.array(y)
		self.labels = np.unique(y)

		if self.fit_intercept==True:
			bias = np.ones((self.X.shape[0], 1))
			self.X = np.append(bias, self.X, axis=1)
		
		self.features = self.X.shape[1]
		self.samples = len(self.X)
		self.coef_= np.ones((self.features,y.shape[1]))

		agrad = elementwise_grad(self.multi_cost)
		for i in range(self.iterations):
			objective = agrad(self.coef_, self.X, self.y)
			self.coef_= self.coef_- self.learning_rate * objective/self.samples
		return self.coef_

	def predict_multi(self,X):
		if self.fit_intercept:
			bias = np.ones((X.shape[0], 1))
			X = np.append(bias, X, axis=1)
		z = 1/(1+np.exp(-(X.dot(self.coef_ ))))		
		y_hat = []
		for i in z:
			y_hat.append(np.argmax(i))
		return y_hat