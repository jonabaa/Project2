#
# This file sets up a neural net and test it on learning on half moon data
# it gives a pretty good accuracy of 97% ! 
#
#

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(1000, noise=0.20)
    return X, y

def visualize(X, y, clf):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary( lambda x: clf.predict(x), X, y)

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title("Logistic Regression")
    plt.show()
    
def classify(X, y):
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    return clf

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

###############
# my stuff

print("Now building neural network to try to solve the problem")

from NeuralNetwork import NeuralNetwork
from functions import *

model = NeuralNetwork()

# set up and compile the net
model.add_layer(10, relu, relu_diff)
model.add_layer(20, relu, relu_diff)
model.add_layer(35, relu, relu_diff)
model.add_layer(20, relu, relu_diff)
model.add_layer(10, relu, relu_diff)
model.add_layer(1, sigmoid, sigmoid_diff)

model.set_cost_function(cost_crossentr, cost_crossentr_diff)

model.set_inputnodes(X_train.shape[1])
model.compile()

# set training rate
model.set_learning_rate(.001)

# reshape y_train
y_train =np.reshape(y_train, (len(y_train), 1))

# train the net
#model.fit_minibatch(X_train, y_train, 10)
model.fit_stochastic_grad(X_train, y_train, iters=1000, batch_size=100)

# check accuracy of net
acc = mean(equals(model.predict_class(X_test), y_test))

print("Training data size: ", X_train.shape[0])
print("Test data size: ", X_test.shape[0])
print("Accuracy: ", acc)


pdb = lambda x : model.predict_class(x)

# Set min and max values and give it some padding
x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
h = 0.01

# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the function value for the whole gid
Z = pdb(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
plt.title("Neural network")
plt.show()

#print("Accuracy: %f" % accuracy_score(y_test, pdb(X_test)))
