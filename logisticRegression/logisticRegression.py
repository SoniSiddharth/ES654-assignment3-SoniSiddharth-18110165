# Importing libraries
import numpy as np
import pandas as pd
import warnings
import autograd.numpy as np
from autograd import elementwise_grad
import autograd as grad
from scipy.special import expit, logit
warnings.filterwarnings( "ignore" )
import matplotlib.pyplot as plt
import matplotlib.colors

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
	# print(lmbda)
	return -np.sum(y*(np.log(z)) + (1-y)*np.log(1 - z)) + lmbda*(theta.T*theta)

class LogisticRegression() :
	def __init__(self, learning_rate=0.001, iterations=1000, lmbda=0.1, regularization=None, fit_intercept=True):
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.lmbda = lmbda
		self.regularization = regularization
		self.fit_intercept = fit_intercept
	
	def fit(self, X, y):
		self.X = np.array(X)	
		self.y = np.array(y)
		
		if self.fit_intercept:
			bias = np.ones((self.X.shape[0], 1))
			self.X = np.append(bias, self.X, axis=1)
		self.rows, self.cols = self.X.shape
		self.coef_= np.zeros(self.cols)
		
		for i in range(self.iterations):
			y_hat = 1/(1 + np.exp(-(self.X.dot(self.coef_))))
			error = (np.array(y_hat) - np.array(self.y.T))	
			error = np.reshape(error, self.rows)
			self.coef_= self.coef_- self.learning_rate * np.dot(self.X.T, error)/self.rows
		return self.coef_

	def fit_autograd(self, X, y):
		self.X = np.array(X)	
		self.y = np.array(y)
		
		if self.fit_intercept:
			bias = np.ones((self.X.shape[0], 1))
			self.X = np.append(bias, self.X, axis=1)
		self.rows, self.cols = self.X.shape
		self.coef_= np.zeros(self.cols)

		if self.regularization==None:
			agrad = elementwise_grad(cost_function)
		elif self.regularization=='l1':
			agrad = elementwise_grad(cost_function_l1)
		elif self.regularization=='l2':
			agrad = elementwise_grad(cost_function_l2)

		for i in range(self.iterations):
			objective = agrad(self.coef_, self.X, self.y, self.lmbda)
			self.coef_= self.coef_- self.learning_rate * objective/self.X.shape[0]
		return self.coef_
	
	def predict(self, X):
		if self.fit_intercept:
			bias = np.ones((X.shape[0], 1))
			X = np.append(bias, X, axis=1)
		z = 1/(1+np.exp(-(X.dot(self.coef_ ))))
		y_hat = np.where(z>0.5, 1, 0)
		return y_hat

	def predict_cross_val(self, X, theta):
		if self.fit_intercept:
			bias = np.ones((X.shape[0], 1))
			X = np.append(bias, X, axis=1)
		z = 1/(1+np.exp(-(X.dot(theta))))		
		y_hat = np.where(z>0.5, 1, 0)
		return y_hat

	def plot_decision_boundary(self, X, y, model):
		bias = self.coef_[0]
		weight1, weight2 = self.coef_[1], self.coef_[2]
		c = -bias/weight2
		m = -weight1/weight2
		xmin, xmax = -1, 2
		ymin, ymax = -1, 2.5
		xd = np.array([xmin, xmax])
		yd = m*xd + c
		plt.plot(xd, yd, 'k', lw=1, ls='--')
		plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
		plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

		plt.scatter(*X[y==0].T, s=8, alpha=0.5)
		plt.scatter(*X[y==1].T, s=8, alpha=0.5)
		plt.xlim(xmin, xmax)
		plt.ylim(ymin, ymax)
		plt.ylabel(r'$x_2$')
		plt.xlabel(r'$x_1$')
		plt.savefig("./logisticRegression/plots/q1_decision_boundary.png")
		plt.show()
