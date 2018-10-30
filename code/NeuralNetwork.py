# This class implements a multilayered neural network
#
# Usage:
# 
# 1. Specify yor network
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

from numpy import random, zeros, sum, floor, around
from numpy.random import choice

class NeuralNetwork:

    def __init__(this):
        this.number_of_nodes = [] # number of nodes in each layer
        this.number_of_inputs = None # number of input nodes

        this.W = [] # list containing all the L weightmatrices
        this.b = [] # list containing all the L biasvectors
        
        this.grad_W = [] # list containing gradients for the weightmatrices
        this.grad_b = [] # list containing gradients for the biasvectors

        this.act = [] # list containing all the L activationfunctions
        # list containing all the L activationfunctin derivatives
        this.act_diff = [] 
        
        this.cost_func = None # The cost function for training the network
        this.cost_func_diff = None # The derivative of the cost function
        
        this.eta = .1 # the learning rate of the network
        
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

        this.W.append((random.random((n[0], inp)) - .5)*.7/.5)
        this.b.append((random.random((n[0], 1)) - .5)*.7/.5)
        
        for l in range(len(this.number_of_nodes)-1):
            this.W.append((random.random((n[l+1],n[l]))-.5)*.7/.5)
            this.b.append((random.random((n[l+1],1)) - .5)*.7/.5)
        
        this.compiled = True



    #############################################
    # Functions for training the neural network #
    #############################################

    
    # Generates evenly sized minibatches
    #
    #@N: Number of elements in total
    #@b: Number of minibatches to generate
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
        

    # Predicts the response/output given a predictor/input
    # Also saves the activations, z, in the network for the given input
    #
    # @x: the input/predictor for the neural network to predict on as
    #     a row-vector
    #
    def forward(this, x):
        if not this.compiled:
            print("Network is not compiled. Use compile().")
            return
        
        L = len(this.b)
        
        # set up list of vectors of activations
        # MIGHT BE SIMPLIFIED SINCE THE ELEMENTS ARE replaced later
        z = []
        for l in range(L):
            z.append(zeros((len(this.b[l]), 1)))
        

        # compute the output
        z[0] = this.W[0].dot(x.T) + this.b[0]
        
        for l in range(1, L):
            z[l] = this.W[l].dot(this.act[l-1](z[l-1])) + this.b[l]

        y = this.act[L - 1](z[L - 1])
        
        return z, y
    
    
    # The much talked about backpropagation
    # 
    # @x: predictor/input (as ROW VECTOR!)
    # @z: list of the activations for each layer in the network given x as
    #     input
    # @t: targets
    #
    def back_propagate(this, x, z, t):
        if not this.compiled:
            print("Network is not compiled. Use compile().")
            return
        
        # setting up neede variables, lists and so on
        L = len(this.W)
        delta = [None]*L
        
        # set up list of gradients with respect to the weights and biases
        gradient_W = [zeros(this.W[l].shape) for l in range(L)]
        gradient_b = [zeros(this.b[l].shape) for l in range(L)]
        
        # Compute delta_L
        delta[L-1] = this.cost_func_diff(this.act[L-1](z[L-1]), t)*this.act_diff[L-1](z[L-1])
        
        # Compute delta_l for l=L-1, L-2, ..., 1
        for l in range(L-2, -1, -1):
            da_dz = this.act_diff[l](z[l])
            delta[l] = this.W[l+1].T.dot(delta[l+1])*this.act_diff[l](z[l])

        # compute the gradient w.r.t. the weights
        for l in range(L-2, 0, -1):
            gradient_W[l] = delta[l]*this.act[l-1](z[l-1]).T
        
        # special treatment of last iteration
        gradient_W[0] = delta[0]*x

        return gradient_W, delta


    # computes the gradient with respect to the weights and biases for a 
    # given sample x 
    def compute_gradient(this, x, t):
        z, y = this.forward(x)
        gradient_W, gradient_b = this.back_propagate(x, z, t)

        return gradient_W, gradient_b


    # Fit yor network using stochastic gradient method
    #
    # @X: trainingdatamatrix, rows are datapoints
    # @T: targets for training data X, rows are datapoints
    # @iters: number of iterations of your "gradientminimizer"
    # @batch_size: size of each random sample from the training data
    #
    def fit_stochastic_grad(this, X, T, epochs=5, batch_size=10):
        if not this.compiled:
            print("Network is not compiled. Use compile().")
            return

        L = len(this.W)

        iters = X.shape[0]//batch_size
        
        # FIX: send the gradient list as argument to compute_gradient

        # set up temporary gradient and weight holders and next weights,grds
        gradient_W = [zeros(this.W[l].shape) for l in range(L)]
        gradient_b = [zeros(this.b[l].shape) for l in range(L)]
        W_next = [zeros(this.W[l].shape) for l in range(L)]
        b_next = [zeros(this.b[l].shape) for l in range(L)]
        
        this.fitted = True
        
        for k in range(epochs):
            # learn network using stochastic gradient method 
            for i in range(iters):
                print("Iteration %s ..." % (i+1))
                
                # pick random numbers from 0 to x.shape 
                batch = choice(X.shape[0], batch_size, replace=False)

                # store initial cost
                cost = 0
                for i in batch:
                    cost -= this.cost_func(this.predict_probability(X[i:i+1,:]), T[i:i+1,:])

                for j in batch:
                    # compute the gradient of the costfunction for sample j 
                    smpl_gradient_W, smpl_gradient_b = this.compute_gradient(X[j:j+1,:], T[j:j+1,:])

                    # add this gradient/N to the total gradient
                    for l in range(L):
                        gradient_W[l] += smpl_gradient_W[l]/batch_size
                        gradient_b[l] += smpl_gradient_b[l]/batch_size

                # update weights and biases
                for l in range(L):
                    W_next[l] = this.W[l] - this.eta*gradient_W[l]
                    this.W[l] = W_next[l]
                    b_next[l] = this.b[l] - this.eta*gradient_b[l]
                    this.b[l] = b_next[l]


                for i in batch:
                    cost += this.cost_func(this.predict_probability(X[i:i+1,:]), T[i:i+1,:])
                print("Cost increment: %f" % cost)
        

    # Fit yor network using minibatches
    #
    # @X: trainingdatamatrix, rows are datapoints
    # @T: targets for training data X, rows are datapoints
    # @number_of_batches: the number of batches to divide the training 
    #                     data into
    #
    def fit_minibatch(this, X, T, number_of_batches):
        if not this.compiled:
            print("Network is not compiled. Use compile(). Fit aborted.")
            return

        if number_of_batches > X.shape[0]:
            print("Number of batches can't be larger than number of training data points. Fit aborted.")
            return

        L = len(this.W)


        
        # FIX: send the gradient list as argument to compute_gradient

        # set up temporary gradient and weight holders and next weights,grds
        gradient_W = [zeros(this.W[l].shape) for l in range(L)]
        gradient_b = [zeros(this.b[l].shape) for l in range(L)]
        W_next = [zeros(this.W[l].shape) for l in range(L)]
        b_next = [zeros(this.b[l].shape) for l in range(L)]

        # set up minibatches
        batches = this.generate_minibatches(X.shape[0], number_of_batches)

        # learn network using minibatch gradient method
        for i in range(len(batches)):
            print("Batch %s/%s ..." % ((i+1), len(batches)))
            
            this.fitted = True
            
            # store initial cost
            cost = 0
            cost -= sum(this.cost_func(this.predict_probability(X[batches[i],:]), T[batches[i],:]))

            for j in batches[i]:
                # compute the gradient of the costfunction for sample j 
                smpl_gradient_W, smpl_gradient_b = this.compute_gradient(X[j:j+1,:], T[j:j+1,:])

                # add this gradient/N to the total gradient
                for l in range(L):
                    gradient_W[l] += smpl_gradient_W[l]
                    gradient_b[l] += smpl_gradient_b[l]

            # update weights and biases
            for l in range(L):
                W_next[l] = this.W[l] - this.eta*gradient_W[l]
                this.W[l] = W_next[l]
                b_next[l] = this.b[l] - this.eta*gradient_b[l]
                this.b[l] = b_next[l]
            
            # compute the increase in cost after this iteration
            # (we prefer this to be a negative number then)
            cost += sum(this.cost_func(this.predict_probability(X[batches[i],:]), T[batches[i],:]))
            print("Cost increment: %f" % cost)
        


####################################################
# Functions for predicting with the neural network #
####################################################


    # Predicts the response/output given a predictor/input
    #
    # @x: the input/predictor for the neural network to predict on as
    #     a row vector
    #
    def predict_probability(this, x):
        if not this.fitted:
            print("Network is not fitted. Use compile().")
            return
        
        y = x.T
        
        # compute the output
        for l in range(len(this.W)):
            y = this.act[l](this.W[l].dot(y) + this.b[l])

        return y


    # Predicts the class given a predictor/input
    #
    # @x: the input/predictor for the neural network to predict on as
    #     a row vector
    #
    def predict_class(this, x):
        if not this.fitted:
            print("Network is not fitted. Use compile().")
            return
        
        y = x.T
        
        # compute the output
        for l in range(len(this.W)):
            y = this.act[l](this.W[l].dot(y) + this.b[l])

        #if y > .5:
        #    return 1
        #else:
        #    return 0
        return around(y)



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


