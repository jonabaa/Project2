from datageneration import *
from NeuralNetworkRegression import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Residual sums squared
def RSS(y, y_tilde):
    return np.sum((y - y_tilde)**2, axis=0)

# Mean squared error
def MSE(y, y_tilde):
    return RSS(y, y_tilde)*(1/len(y))

# R2-score function
def R2Score(y, y_tilde):
    return 1 - RSS(y, y_tilde)/np.sum((y - np.sum(y, axis=0)/y.size)**2, axis=0)

# generate ising data: One dimension, L=40, N=100000
states, energies = generate_data(40, 100000)

print(states.shape)
print(energies.shape)

# split into test and training data: 20% test data
X_train, X_test, y_train, y_test = train_test_split(
        states, energies, test_size=.2)

# reshape to give y_train and y_test 2 dimensions
y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

sizes = [10**i for i in range(2, 4)]
lambdas = [10**i for i in range(-2, -6, -1)]

mse = np.zeros((len(sizes), len(lambdas)))
r2 = np.zeros((len(sizes), len(lambdas)))

for i in range(len(sizes)):
    print("N = %s" % sizes[i])
    # split into test and training data: 20% test data
    X_train, X_test, y_train, y_test = train_test_split(
            states[:sizes[i],:], energies[:sizes[i],:], test_size=.2)

    # reshape to give y_train and y_test 2 dimensions
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # set up neural network
    model = NeuralNetwork(X_train, y_train)
    
    for j in range(len(lambdas)):
        model.lmbd = lambdas[j]
        
        # train neural network
        model.train()

        mse[i,j] = MSE(y_test, model.predict(X_test))
        r2[i,j] = R2Score(y_test, model.predict(X_test))
        print("MSE: ", MSE(y_test, model.predict(X_test)))
        print("R2-score: ", R2Score(y_test, model.predict(X_test)))



import matplotlib.pyplot as plt
import seaborn as sns


# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(r2, annot=True, fmt="f", linewidths=.5, ax=ax)

