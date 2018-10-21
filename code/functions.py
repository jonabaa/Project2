# This module contains cost functions, activation functions and their
# derivatives.

from numpy import exp, sum


#
# Activation functions
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


# The derivative of the cost function for a single sample with respect to
# y_i
def cost_diff(y, t, i):
    return y[i] - t[i]


# Computes the cost over all samples and all responses
def total_cost(Y, T):
    return sum(cost(Y, T), axis=0)

