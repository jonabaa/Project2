from project1.OLSLinearModel import OLSLinearModel
from project1.RidgeLinearModel import RidgeLinearModel
from project1.LassoLinearModel import LassoLinearModel
import numpy as np

# Bootstrap with B resamples
#
# @X: inputdata
# @y: outputdata
# @B: number of bootstrap-samples
# @model: the method for fitting model (OLS, ridge or lasso)
# @lmb: lambda (set to 0 for OLS-regression)
#
def Bootstrap(X, y, B, model, lmb=.1):
    N = X.shape[0]
    
    if model == "ols":
        print("Bootstrapping (OLS) ...")
        model = OLSLinearModel()
    elif model == "ridge":
        print("Bootstrapping (Ridge) ...")
        model = RidgeLinearModel(lmb)
    elif model == "lasso":
        print("Bootstrapping (Lasso) ...")
        model = LassoLinearModel(lmb)
    else:
        print("Invalid model. Valid choices: ols, ridge or lasso.")
        return

    # set up design matrix 
    X=np.einsum('...i,...j->...ij', X, X)
    shape=X.shape
    X=X.reshape((shape[0],shape[1]*shape[2]))

    for b in range(B):
        print("Batch %s of %s " % (b+1, B))
        
        # draw a random bootstrap batch with replacement
        batch = np.random.choice(N, N)
        
        # fit model to bootstrap sample
        model.fit(X[batch], y[batch])

        # compute and store y_tilde (prediction of model give X)
        y_tilde = model.predict(X[batch])
        
        if b == 0:
            y_tilde_matrix = y_tilde
        else:
            y_tilde_matrix = np.concatenate([y_tilde_matrix, y_tilde], axis=1)
    
    #compute expected value in each x over the bootstrapsamples
    E_L = (np.mean(y_tilde_matrix, axis=1, keepdims=True))

    # compute bias
    bias = np.mean((y - E_L)**2)

    # compute variance
    var = np.mean(np.mean((y_tilde_matrix - E_L)**2, axis=1, keepdims=True))

    # do some printing for test purposes
    print("VAR: %f" % var)
    print("BIAS: %f" % bias)

    return bias, var


