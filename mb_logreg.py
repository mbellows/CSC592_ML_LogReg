import pandas as pd
import numpy as np
from math import exp, log
from scipy import special as sp


learningRate = 1

#Sigmoid function for whole training set.
def sigmoid (X, theta):

    return sp.expit(-(np.dot(X, theta)))
    #return 1 / (1 + np.exp(-(np.dot(X, theta))))


#Finds hypothesis given training set X and labels Y. Returns the hypothesis
def findHypothesis(X, Y):

    theta = pd.DataFrame(data = np.zeros((27, 1)))          #Setting initial theta/weights to zero
    theta = theta - (learningRate / 13000) * (np.dot(X.T, (sigmoid(X, theta) - Y)))
    return theta.astype('float')


def cost(X, Y, theta):

    df = pd.DataFrame(-(1 / 13000) * np.dot(Y.T, np.log(sigmoid(X, theta))))
    df.to_csv('debug.csv',sep=",",float_format='%10.8f',index=False)
    #return (1 / 13000) * (np.dot(-Y.T, np.log(sigmoid(X, theta))) - np.dot((1 - Y).T, np.log(1 - sigmoid(X, theta))))
    return -(1 / 13000) * np.dot(Y.T, np.log(sigmoid(X, theta)))