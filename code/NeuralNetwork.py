# This class implements a multilayered neural network
#
# Usage:
# 
# 1. Specify your network
#    a) Set number of inout nodes
#    b) Add layers
#    c) Set costfunction
#    d) compile
# 
# 2. Train the network
#
# 3. Predict with your network
#
#

from numpy import random, zeros, sum, floor, around, mean
from numpy.random import choice
from functions import equals
from sklearn.metrics import accuracy_score

class NeuralNetwork:

    def __init__(this):
        this.number_of_nodes = [] # number of nodes in each layer
        this.number_of_inputs = None # number of input nodes

        this.W = [] # list containing all the L weightmatrices
        this.b = [] # list containing all the L biasvectors
        
        this.act = [] # list containing all the L activationfunctions
        # list containing all the L activationfunctin derivatives
        this.act_diff = []
        
        this.cost_func = None # The cost function for training the network
        this.cost_func_diff = None # The derivative of the cost function
        
        this.eta = .1 # the learning rate of the network
        this.lmb = .1 # the regularization parameter
        this.descent_rate_eta = 1 # descent rate of learning rate

        # flag determining wether the network can start to learn
        this.compiled = False 
        # flag determining wether the network can predict
        this.fitted = False 
        

    ###############################################
    # Functions for setting up the neural network #
    ###############################################

    
    # Add a layer to the end of the network
    #
    # @nodes: The number of nodes in this layer
    # @act_func: The activation function of this layer
    # @act_func_diff: The derivative of the activation function 
    #                 of this layer
    #
    def add_layer(this, nodes, act_func, act_func_diff):
        # store the number of nodes in the layer
        this.number_of_nodes.append(nodes)
        # store the activation function of the layer
        this.act.append(act_func)
        # store the derivative of the activation function of the layer
        this.act_diff.append(act_func_diff)

    
    # Set the number of input nodes in the network
    def set_inputnodes(this, n):
        this.number_of_inputs = n


    # Set the cost function for the network
    def set_cost_function(this, cost_func, cost_func_diff):
        this.cost_func = cost_func
        this.cost_func_diff = cost_func_diff

    
    # Set the learning rate of the network
    def set_learning_rate(this, eta):
        this.eta = eta # the learning rate of the network

    
    # Set the descent rate of the learning rate for each epoch
    def set_descent_rate(this, x):
        this.descent_rate = x


    # Compile the network
    # This function sets up the network so that it is ready to learn 
    # from a given set of training data. Must be run after the network 
    # is specified and before training. 
    def compile(this):
        if (len(this.number_of_nodes) == 0):
            print("No layers. Compilation unsuccessfull.")
            return

        if (this.cost_func == None):
            print("Cost function is missing. Compilation unsuccessfull.")
            return

        if (this.cost_func_diff == None):
            print("Cost function derivative is missing. Compilation unsuccessfull.")
            return
        
        if (this.number_of_inputs == None):
            print("Number of inputnodes is missing. Compilation unsuccessfull.")
            return

        # set up the weights and biases of the layers
        n = this.number_of_nodes
        inp = this.number_of_inputs

        L = len(this.number_of_nodes)

        this.W.append(random.randn(n[0], inp)*.2)
        this.b.append(random.randn(n[0], 1)*.2)
        
        for l in range(L-1):
            this.W.append(random.randn(n[l+1],n[l])*.2)
            this.b.append(random.randn(n[l+1],1)*.2)
        
        # To save time on memory allocation and possibly memory 
        # we let the following variables be public inside this class:
        #
        # set up list of gradients with respect to the weights and biases
        # for each layer
        #this.gradient_W = [zeros(this.W[l].shape) for l in range(L)]
        #this.gradient_b = [zeros(this.b[l].shape) for l in range(L)]
        #this.delta = [None]*L
        
        this.compiled = True



    #############################################
    # Functions for training the neural network #
    #############################################

    
    # Generates evenly sized minibatches
    #
    # INPUT
    # @N: Number of elements in total
    # @b: Number of minibatches to generate
    #
    # OUTPUT
    # @p1 + p2: a list of lists containing the indicies of the inputs
    #           in each batch
    #
    def generate_minibatches(this, N, b):
        if b == 0:
            print("Error: Number of batches must be positive.")
            return

        if N <= 0:
            print("Error: Number of elements must be positive.")
            return

        p1 = [list(range( i*(N//b + 1), (i + 1)*(N//b + 1) )) for i in range(N%b)]
        p2 = [list(range( (N%b)*(N//b + 1) + i*(N//b), (N%b)*(N//b + 1) + (i + 1)*(N//b) )) for i in range(b - N%b)]
        
        return p1 + p2
        

    # Computes the activations of the network for the given input,
    # and the resulting output
    #
    # INPUT
    # @x: the input/predictor for the neural network to predict on as
    #     a row-vector
    #
    # OUTPUT
    # @z: list of column-vectors containing the activations in each layer
    # @y: the output of the network
    #
    def forward(this, x):
        if not this.compiled:
            print("Network is not compiled. Use compile().")
            return
        
        L = len(this.b)
        
        # set up list of vectors for activations in each layer
        z = [None]*L

        # compute the activations
        z[0] = this.W[0].dot(x.T) + this.b[0]
        
        for l in range(1, L):
            z[l] = this.W[l].dot(this.act[l-1](z[l-1])) + this.b[l]

        # compute the output
        y = this.act[L - 1](z[L - 1])
        
        return z, y
    
    
    # The backpropagation algorithm. Computes the derivatives of all 
    # weights and biases in all layers of network for the input x and
    # ADDS this to the lists delta and gradient_W
    # 
    # INPUT
    # @x: predictor/input as row-vector
    # @z: list of the activations (as row vectors) for each layer in the 
    #     network given x as input
    # @y: output of the network given x as input
    # @t: targets
    # @delta: list of deltas (as column vectors) for each layer in network 
    # @gradient_W: list of weight derivative matrices for each layer in
    #              the network
    #
    def back_propagate(this, x, z, y, t, delta, gradient_W):
        #if not this.compiled:
        #    print("Network is not compiled. Use compile().")
        #    return
        
        # setting up neede variables, lists and so on
        L = len(this.W)
        
        # Compute delta_L-1
        delta[L-1] += this.cost_func_diff(this.act[L-1](z[L-1]), t)*this.act_diff[L-1](z[L-1])
        
        # Compute delta_l for l=L-2, L-3, ..., 0
        for l in range(L-2, -1, -1):
            da_dz = this.act_diff[l](z[l])
            delta[l] += this.W[l+1].T.dot(delta[l+1])*da_dz

        # compute the gradient w.r.t. the weights
        for l in range(1, L):
            gradient_W[l] += delta[l]*this.act[l-1](z[l-1]).T + 2*this.lmb*this.W[l]
        
        # special treatment of last iteration
        gradient_W[0] += delta[0]*x


    # computes the gradient with respect to the weights and biases for a 
    # given sample x 
    def compute_gradient(this, x, t, delta, gradient_W):
        
        z, y = this.forward(x)

        this.back_propagate(x, z, y, t, delta, gradient_W)


    # Fit yor network using stochastic gradient method
    #
    # @X: trainingdatamatrix, rows are datapoints
    # @T: targets for training data X, rows are datapoints
    # @epochs: number of epochs of your "gradientminimizer"
    # @batches: a list of lists determining which samples to compute 
    #           gradients for for each iteration
    #
    def fit(this, X, T, batches, epochs):
        if not this.compiled:
            print("Network is not compiled. Use compile(). Fit aborted.")
            return

        L = len(this.W)

        this.fitted = True
        
        # store initial cost
        cost_init = sum(this.cost_func(this.predict_probability(X), T))
        
        # The main loop for training the network
        print("Training network ...")

        for epoch in range(epochs):
            print("Epoch %s / %s ... " % (epoch + 1, epochs))
            
            for batch in batches:
                # set up storage for derivatives
                delta = [zeros(this.b[l].shape) for l in range(L)]
                gradient_W = [zeros(this.W[l].shape) for l in range(L)]

                for j in batch:
                    # compute the gradient of the costfunction for sample j 
                    this.compute_gradient(X[j:j+1,:], T[j:j+1,:], delta, gradient_W)

                # update weights and biases
                for l in range(L):
                    this.W[l] += this.eta*gradient_W[l]/len(batch)
                    this.b[l] += this.eta*delta[l]/len(batch)
            
            # compute and print accuracy on test data
            #print(this.predict_probability(X[1:2,:]))
            acc = mean(equals(this.predict_class(X), T))            
            print("Accuracy on training data: %f " % acc)

            this.eta *= this.descent_rate_eta


    # Fit yor network using stochastic gradient method
    #
    # @X: trainingdatamatrix, rows are datapoints
    # @T: targets for training data X, rows are datapoints
    # @epochs: number of iterations of your "gradientminimizer"
    # @batch_size: size of each random sample from the training data
    #
    def fit_stoc_batch(this, X, T, batch_size, epochs=5):
        # compute number of random batches to be made
        B = X.shape[0]//batch_size
        
        # create B random batches
        batches = [choice(X.shape[0], batch_size, replace=False) for i in range(B)]

        this.fit(X, T, batches, epochs)


    # Fit your network using minibatches
    #
    # @X: trainingdatamatrix, rows are datapoints
    # @T: targets for training data X, rows are datapoints
    # @number_of_batches: the number of batches to divide the training 
    #                     data into
    # @epochs: number of iterations of your "gradientminimizer"
    #
    def fit_det_batch(this, X, T, number_of_batches, epochs=5):
        # create number_of_batches evenly sized batches
        batches = this.generate_minibatches(X.shape[0], number_of_batches)
        

        this.fit(X, T, batches, epochs)



####################################################
# Functions for predicting with the neural network #
####################################################


    # Predicts the response/output given a predictor/input
    # 
    # INPUT
    # @x: the input/predictor for the neural network to predict on as
    #     a row vector
    # 
    # OUTPUT
    # @y.T: the prediction of the network as a row vector
    #
    def predict_probability(this, x):
        #if not this.fitted:
        #    print("Network is not fitted. Use compile().")
        #    return
        
        y = x.T
        
        # compute the output
        for l in range(len(this.W)):
            y = this.act[l](this.W[l].dot(y) + this.b[l])

        return y.T


    # Predicts the class given a predictor/input
    #
    # @x: the input/predictor for the neural network to predict on as
    #     a row vector
    #
    def predict_class(this, x):
        #if not this.fitted:
        #    print("Network is not fitted. Use compile().")
        #    return
        
        return around(this.predict_probability(x))



################
# Test program #
################

"""
# test program

from numpy import random, ones
from functions import *

N = 100
p = 3
iterations = 50

model = NeuralNetwork()

# set up sample matrix
X = random.rand(N,p)

# set up targets/responses
y = random.choice([0,1], N)
y = y.reshape((N,1))

# set up and compile the net
model.add_layer(5, sigmoid, sigmoid_diff)
model.add_layer(1, sigmoid, sigmoid_diff)
model.set_cost_function(cost, cost_diff)
model.set_inputnodes(p)
model.compile()

# train the net
model.fit(X, y, iters=5, batch_size=10)

# predict with the net
x = ones((1, 3))
print(model.predict(x))
"""


