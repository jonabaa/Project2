# This module contains cost functions, activation functions and their
# derivatives.

import numpy as np
from numpy import exp, sum, mean, zeros


#
# Activation functions and their derivatives
#


# the sigmoid activation function
def sigmoid(x):
    return exp(x)/(1 + exp(x))


# the derivative of the sigmoid activation function
def sigmoid_diff(x):
    return sigmoid(x)*(1 - sigmoid(x))


#
# Cost functions
#


# Computes the cost for a single sample
#
# @y: a row-vector containing the prediction for the given predictor
# @t: a row-vector containing the target for the given predictor
#
def cost(y, t):
    return .5*sum((y - t)**2, axis=1)


# The derivative of the cost function for a single sample
def cost_diff(y, t):
    return y - t


# Computes the cost over all samples and all responses
def total_cost(Y, T):
    return sum(mean(Y, T), axis=0)


#
# Functions for doing linear regression
#


# Setting up the design matrix for predicting energy of system based on
# spins
#
# @X: the data matrix (samples along 1st axis, predictors along 2nd axis)
#
def compute_design_matrix(X):
    # allocate the design matrix
    Xd = ones((X.shape[0], x.shape[1]**2))
    
    # compute the design matrix
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            Xd[:, X.shape[1]*i + j] *= X[:,i]*X[:,j]

    return Xd


