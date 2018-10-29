# this class implements a logistic regrssion model
from numpy import zeros, ones, exp, concatenate
from numpy.linalg import pinv


class LogisticRegressionModel:
    # constructor

    def __init__(this):
        this.beta = None
    

    # sets up the design matrix 
    #
    # @X: a N by p matrix containing the N samples as rows
    # 
    def design_matrix(this, X):
        return concatenate([ones((X.shape[0],1)), X], axis=1)
    

    # probability for class 1
    #
    # @x: a single sample as a row vector with a preceding 1
    # @beta: the coefficients as a column vector
    #
    def pr(this, x, beta):
        y = exp(x.dot(beta))/(1 + exp(x.dot(beta)))
        return y

    # fits the model using the Newton Raphson method
    #
    # @X: the N by p+1 matrix containing the predictors with ones appended
    #     as the first element
    # @y: the N responses
    # @iterations: number of iterations of the Newton Raphson method
    #
    def fit(this, X, y, iterations):
        print("Fitting ...")
        beta = zeros((X.shape[1],1))
        beta_new = zeros((X.shape[1],1))
        
        # iterative minimization of costfuntion
        for i in range(iterations):
            print("Iteration %s" % (i+1))
            N = X.shape[0]

            # compute p and W
            p = zeros((N,1))
            W = zeros((N,N))

            for j in range(N):
                p[j] = this.pr(X[j,:], beta)
                W[j,j] = p[j]*(1 - p[j])
            
            # update beta
            beta_new = beta + pinv(X.T.dot(W).dot(X)).dot(X.T.dot(y-p))
            beta = beta_new
        
        this.beta0 = beta[0,0]
        this.betas = beta[1:,0]

        return beta


    # probability for class 1
    # probabilty for class two is 1 - probability for class 1
    #
    # @x: a single sample as a row vector
    #
    def predict_probability(this, x):
        return exp(this.beta0 + x.dot(this.betas))/(1 
                + exp(this.beta0 + x.dot(this.betas)))

    
    # predict class 1
    def predict_class(this, x):
        if .5 < exp(this.beta0 + x.dot(this.betas))/(1 
                + exp(this.beta0 + x.dot(this.betas))):
            return 1
        else:
            return 0


"""
# test program
from numpy import random

N = 100
p = 3
iterations = 50

model = LogisticRegressionModel()

# set up design matrix
X = random.rand(N,p)
Xd = model.design_matrix(X)

# set up targets/responses
y = random.choice([0,1], N)
y = y.reshape((N,1))

print(model.fit(X, y, iterations))
"""

            
