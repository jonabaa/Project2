from project1.utilities import RSS, MSE, R2Score
import numpy as np
import scipy.stats as st


class RidgeLinearModel:
    covariance_matrix = None # covariance matrix of the model coefficients
    covariance_matrix_updated = False
    beta = None # coefficients of the modelfunction
    var_vector = None
    var_vector_updated = False
    CIbeta = None # confidence interval of betas
    CIbeta_updated = False
    x1 = None # first predictor of sampledata
    x2 = None # second predictor of sampledata
    y = None # responses of sampledata
    y_tilde = None # model predictions for x
    y_tilde_updated = False


    def __init__(this, lmb=.01):
        this.lmb = lmb


    def set_params(this, alpha=.1):
        this.lmb = alpha
    

    # This function sets up the design matrix for our Ising problem
    #
    # @X: N x p matrix containing the input
    # 
    def design(this, X):
        parts = [X*X[:][i:i+1] for i in range(X.shape[1])]
        return -np.concatenate(parts, axis=1)

    
    # This function fits the model to your data
    #
    # @X: The design matrix
    # @y: The targets/output
    #
    def fit(this, X, y):
        # compute linear regression coefficients
        this.beta = np.linalg.pinv(X.T.dot(X) +
                this.lmb*np.identity(X.shape[1])).dot(X.T).dot(y)

        this.coef_ = this.beta


    # Predicts and returns the responses of the predictors with
    # the fitted model
    #
    # @X: N x p matrix containing the input
    #
    def predict(this, X):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            #parts = [X*X[:,i:i+1] for i in range(X.shape[1])]
            #X = np.concatenate(parts, axis=1)

            return X.dot(this.beta)

    
    def get_beta(this):
        return this.beta


    # Returns the R2-score of the model on the given data
    #
    # @X: input
    # @y: output
    #
    def score(this, X, y):
        return R2Score(y, this.predict(X))



    #############
    # Old stuff #
    #############

"""
    # Returns the residuals of the model squared and summed
    def get_RSS(this, x1, x2, y):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            y_tilde = this.predict(x1, x2)
            return RSS(y, this.y_tilde)


    # Returns the mean squared error of the model
    # given the sample data (x1, x2, y)
    #
    # @x1: vector of first predictor
    # @x2: vector of second predictor
    # @y: vector of responses
    #
    def get_MSE(this, x1, x2, y):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            y_tilde = this.predict(x1, x2)
            return MSE(y, y_tilde)


    # Returns the R2 score of the model
    def get_R2Score(this, x1, x2, y):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            y_tilde = this.predict(x1, x2)
            return R2Score(y, y_tilde)


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
            # Next line fixes if y is one-dimensional
            if (len(s_y.shape)) == 1:
                s_y = np.expand_dims(this.y[c], axis=1)

            # allocate design matrix
            s_X = np.ones((m, n))

            # compute values of design matrix
            for i in range(m): # vectoriser denne løkka
                for p in range(this.k):
                    for j in range(SumOneToN(p + 2) - SumOneToN(p + 1)):
                        s_X[i][SumOneToN(p + 1) + j] *= s_x1[i]**(p
                                + 1 - j)*s_x2[i]**j

            betasamples[:,b] = np.linalg.pinv(s_X.T.dot(s_X) +
                    this.lmb*np.identity(n)).dot(s_X.T).dot(s_y)[:, 0]

        betameans = betasamples.sum(axis=1, keepdims=True)/B

        # Compute variance vector
        this.var_vector = np.sum((betasamples - betameans)**2, axis=1)/B

        return this.var_vector


    # Returns the confidence interval of the betas
    def get_CI_of_beta(this, percentile=.95):
        if this.beta is None:
            print("Error: Model is not fitted.")
            return None
        else:
            if not this.CIbeta_updated:

                # stdcoeff is the z-score to the two-sided confidence interval
                stdcoeff = st.norm.ppf((1-percentile)/2)
                this.CI_beta = np.zeros((len(this.beta), 2))
                for i in range(len(this.beta)):
                    this.CI_beta[i][0] = this.beta[i] + stdcoeff*np.sqrt(this.var_vector[i])
                    this.CI_beta[i][1] = this.beta[i] - stdcoeff*np.sqrt(this.var_vector[i])

                this.CIbeta_updated = True
                # CI_beta returns a nx2 matrix with each row
                # representing the confidence interval to the corresponding beta
            return this.CI_beta


    def set_updated_to_false(this):
        covariance_matrix_updated = False
        var_vector_updated = False
        y_tilde_updated = False
        CIbeta_updated = False

"""
