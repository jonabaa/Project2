from project1.RidgeLinearModel import RidgeLinearModel
from project1.OLSLinearModel import OLSLinearModel
from datageneration import *
from project1.utilities import RSS, MSE, R2Score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn


L = 40 # set number of predictors for each sample
N = 2000 # set number of samples

# generate data
states, energies = generate_data(L, N)

# set up design matrix
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))

# splitting into training- and testdata
X_train, X_test, Y_train, Y_test = train_test_split(states, energies, train_size= .4)

# set up Lasso and Ridge Regression models
leastsq = OLSLinearModel()
ridge = RidgeLinearModel()
lasso = linear_model.Lasso()

# set up error lists
train_errors_leastsq = []
test_errors_leastsq = []

train_errors_ridge = []
test_errors_ridge = []

train_errors_lasso = []
test_errors_lasso = []

# set up list of lambdas
lmbdas = np.logspace(-4, 5, 10)

# Initialize coeffficients for ridge regression and Lasso
coefs_leastsq = []
coefs_ridge = []
coefs_lasso=[]

# settting up lists of coefficient matrices
J_matrix_leastsq = []
J_matrix_ridge = []
J_matrix_lasso = []

for lmbda in lmbdas:
    
    # OLS regression
    leastsq.fit(X_train, Y_train) # fit model 
    coefs_leastsq.append(leastsq.get_beta()) # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_leastsq.append(leastsq.score(X_train, Y_train))
    test_errors_leastsq.append(leastsq.score(X_test,Y_test))
    
    # Ridge regression
    ridge.set_lmb(lmbda) # set regularisation parameter
    ridge.fit(X_train, Y_train) # fit model 
    coefs_ridge.append(ridge.get_beta()) # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_ridge.append(ridge.score(X_train, Y_train))
    test_errors_ridge.append(ridge.score(X_test,Y_test))
    
    # Lasso regression
    lasso.set_params(alpha=lmbda) # set regularisation parameter
    lasso.fit(X_train, Y_train) # fit model
    coefs_lasso.append(lasso.coef_) # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_lasso.append(lasso.score(X_train, Y_train))
    test_errors_lasso.append(lasso.score(X_test,Y_test))
    
    
    
    ### plot Ising interaction J
    J_leastsq=np.array(leastsq.coef_).reshape((L,L))
    J_ridge=np.array(ridge.coef_).reshape((L,L))
    J_lasso=np.array(lasso.coef_).reshape((L,L))

    J_matrix_leastsq.append(J_leastsq)
    J_matrix_ridge.append(J_ridge)
    J_matrix_lasso.append(J_lasso)
    
    """
    cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

    fig, axarr = plt.subplots(nrows=1, ncols=3)
    
    axarr[0].imshow(J_leastsq,**cmap_args)
    axarr[0].set_title('$\\mathrm{OLS}$',fontsize=16)
    axarr[0].tick_params(labelsize=16)
    
    axarr[1].imshow(J_ridge,**cmap_args)
    axarr[1].set_title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(lmbda),fontsize=16)
    axarr[1].tick_params(labelsize=16)
    
    im=axarr[2].imshow(J_lasso,**cmap_args)
    axarr[2].set_title('$\\mathrm{LASSO},\ \\lambda=%.4f$' %(lmbda),fontsize=16)
    axarr[2].tick_params(labelsize=16)
    
    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=fig.colorbar(im, cax=cax)
    
    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
    cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)
    
    fig.subplots_adjust(right=2.0)
    
    plt.show()
    """

# Compute error in J's
# set up true J

J=np.zeros((L,L),)
for i in range(L):
    J[i,(i+1)%L]-=1.0

mse_coefs_OLS = [np.mean((J_matrix_leastsq[i] - J)**2) 
        for i in range(len(J_matrix_leastsq))]
mse_coefs_ridge = [np.mean((J_matrix_ridge[i] - J)**2) 
        for i in range(len(J_matrix_ridge))]
mse_coefs_lasso = [np.mean((J_matrix_lasso[i] - J)**2) 
        for i in range(len(J_matrix_lasso))]



# Plot errors in J's
plt.semilogx(lmbdas, mse_coefs_OLS, 'b',label='OLS')
plt.semilogx(lmbdas, mse_coefs_ridge, 'r',label='Ridge')
plt.semilogx(lmbdas, mse_coefs_lasso, 'g',label='Lasso')

fig = plt.gcf()
plt.title("MSE in coupling constant J")
plt.legend(loc='lower left',fontsize=16)
plt.xlim([min(lmbdas), max(lmbdas)])
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.savefig("MSE_J_N2000_train4_test6.png")
plt.show()


# Plot R2-score on both the training and test data
plt.semilogx(lmbdas, train_errors_leastsq, 'b',label='Train (OLS)')
plt.semilogx(lmbdas, test_errors_leastsq,'--b',label='Test (OLS)')
plt.semilogx(lmbdas, train_errors_ridge,'r',label='Train (Ridge)',linewidth=1)
plt.semilogx(lmbdas, test_errors_ridge,'--r',label='Test (Ridge)',linewidth=1)
plt.semilogx(lmbdas, train_errors_lasso, 'g',label='Train (LASSO)')
plt.semilogx(lmbdas, test_errors_lasso, '--g',label='Test (LASSO)')

fig = plt.gcf()
fig.set_size_inches(10.0, 6.0)

#plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
#           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left',fontsize=16)
plt.ylim([-0.01, 1.01])
plt.xlim([min(lmbdas), max(lmbdas)])
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel('R2Score',fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig("R2_score_N2000_train4_test6.png")
plt.show()


"""
# make instance of ridge linear model
model = RidgeLinearModel(lmb=0.1)

# set up design matrix
Xd =  model.design(X_train)
print( Xd)
print(Xd.shape)
# fit the model
model.fit(Xd, y_train)
model_lasso = Lasso()
model_lasso.fit(Xd, y_train)

# assess
y_tilde = model.predict(X_test)
y_tilde_lasso = model_lasso.predict(X_test)
mse = MSE(y_test, y_tilde)
r2 = R2Score(y_test, y_tilde)

print("MSE: %f" % mse) 
print("R2Score: %f" % r2) 

J = model.get_beta()
mJ = np.zeros((L,L))

for i in range(L):
    for j in range(L):
        mJ[i][j] = J[i*L + j]

print('\a')


#################
# Visualization #
#################

# visual representation of grid search
# uses seaborn heatmap, you can also do this with matplotlib imshow
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

fig, ax = plt.subplots(figsize = (L, L))
sns.heatmap(mJ, annot=False, ax=ax, cmap="viridis")
ax.set_title("Coefficients: J")
ax.set_ylabel("x_1")
ax.set_xlabel("x_2")
plt.show()
"""

