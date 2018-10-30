# This module contains cost functions, activation functions and their
# derivatives.

import numpy as np
from numpy import exp, sum, mean, zeros, heaviside, log


#
# Activation functions and their derivatives
#


# Sigmoid activation function
def sigmoid(x):
    return exp(x)/(1 + exp(x))


# Derivative of the sigmoid activation function
def sigmoid_diff(x):
    return sigmoid(x)*(1 - sigmoid(x))


# RELU activation function
def relu(x):
    return heaviside(x, 0)*x

# RELU derivative, activation function
def relu_diff(x):
    return heaviside(x, 0)


#
# Cost functions
#


# Computes the cost for a single sample
#
# @y: a row-vector containing the prediction for the given predictor
# @t: a row-vector containing the target for the given predictor
#
def cost_lin(y, t):
    return .5*sum((y - t)**2, axis=1)


# The derivative of the cost function for a single sample w.r.t. the output
# from the last layer
def cost_lin_diff(y, t):
    return y - t 


# Computes the cost over all samples and all responses
def total_cost_lin(Y, T):
    return sum(mean(Y, T), axis=0)


# Cross entropy in a single sample, cost function
#
# @y: prediction of neural network
# @t: the target 
#
def cost_crossentr(y, t):
    return  - t*log(y) - (1-t)*log(1-y)


# Cross entropy derivative (w.r.t. output of network), cost function
#
#
# @y: prediction of neural network
# @t: the target 
#
def cost_crossentr_diff(y, t):
    return (y-t)/(y*(1-y))


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


#
# Functions for evaluation of models
#


def equals(x, y):
    return (x == y)*1


