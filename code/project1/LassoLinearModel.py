from project1.RidgeLinearModel import RidgeLinearModel
from sklearn import linear_model
from project1.utilities import *
import numpy as np

class LassoLinearModel(RidgeLinearModel):
    def fit(this, X, y):
        lasso = linear_model.Lasso(alpha=this.lmb)
        lasso.fit(X, y)
        this.beta = np.reshape(lasso.coef_, (len(lasso.coef_), 1))

    # Computes the sample variance of the coefficients of the model
    # @B: The number of samples used
    def get_variance_of_betas(this, B=20):
        m = len(this.x1)
        n = SumOneToN(this.k + 1)
        betasamples = np.zeros((n, B))

        for b in range(B):
            # create bootstrapsample
            c = np.random.choice(len(this.x1), len(this.x1))
            s_x1 = this.x1[c]
            s_x2 = this.x2[c]
            s_y = this.y[c]

            # allocate design matrix
            s_X = np.ones((m, n))

            # compute values of design matrix
            for i in range(m): # vectoriser denne løkka
                for p in range(this.k):
                    for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                        s_X[i][SumOneToN(p + 1) + j] *= s_x1[i]**(p
                                + 1 - j)*s_x2[i]**j

            lasso = linear_model.Lasso(alpha=this.lmb)
            lasso.fit(s_X, s_y)
            betasamples[:,b] = lasso.coef_

        betameans = betasamples.sum(axis=1, keepdims=True)/B

        # Compute variance vector
        this.var_vector = np.sum((betasamples - betameans)**2, axis=1)/B

        return this.var_vector
