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

X, y1 = generate_data()
y1 = np.reshape(y1, (len(y1), 1))
y2 = abs(1 -y1)
y = np.concatenate([y1, y2], axis = 1)
print(y.shape)

###############
# my stuff

print("Now building neural network to try to solve the problem")

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# divide into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2)

# specify the neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(12, activation='relu', input_dim=X.shape[1]))
model.add(tf.keras.layers.Dense(6, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='relu'))
model.add(tf.keras.layers.Dense(y.shape[1], activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=[X_test, y_test])

# print accuracy and save model
print('accuracy:', model.evaluate(X_test, y_test))
#model.save('tictacNET.h5')

pdb = lambda x : model.predict(x)

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
