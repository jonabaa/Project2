import numpy as np

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

np.random.seed(1345) # shuffle random seed generator

# Ising model parameters
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit



##### prepare training and test data sets

import pickle,os
from sklearn.model_selection import train_test_split

###### define ML parameters
num_classes=2
train_to_test_ratio=0.5 # training samples

# path to data directory
path_to_data=os.path.expanduser('~')+'/fys-stk4155/Project2/data/'

# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
print("Loading sampledata - predictors ...")
data = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
print("Processing predictors ...")
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
print("Loading sampledata - responses...")
labels = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

print("Formating data ...")
print()

"""
# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]
"""

# divide data into ordered, critical and disordered
size = 5000

X_ordered=data[:size,:]
Y_ordered=labels[:size]

X_critical=data[70000:70000+size,:]
Y_critical=labels[70000:70000+size]

X_disordered=data[100000:100000+size,:]
Y_disordered=labels[100000:100000+size]

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))
# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

# full data set
X=np.concatenate((X_critical,X))
Y=np.concatenate((Y_critical,Y))

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)

print()

print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

# Testing neural net

from NeuralNetwork import NeuralNetwork
from functions import *

model = NeuralNetwork()

# set up and compile the net
model.add_layer(20, sigmoid, sigmoid_diff)
model.add_layer(1, sigmoid, sigmoid_diff)
model.set_cost_function(cost_crossentr, cost_crossentr_diff)
model.set_inputnodes(X_train.shape[1])
model.compile()

# reshape Y_train and Y_test
Y_train = np.reshape(Y_train, (len(Y_train), 1))
Y_test = np.reshape(Y_test, (len(Y_test), 1))

# set training rate
model.set_learning_rate(.01)

# train the net
# 15 epochs
# batch size 100 
model.fit_stoc_batch(X_train, Y_train, epochs=10, batch_size=100)

# predict and compute accuracy of net
acc1 = mean(equals(model.predict_class(X_test), Y_test))
acc2 = mean(equals(model.predict_class(X_train), Y_train))

print("Accuracy on test data:", acc1)
print("Accuracy on training data:", acc2)

