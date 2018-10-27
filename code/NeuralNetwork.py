# This class implements a multilayered neural network

from numpy import random, zeros, sum, random.choice

class NeuralNetwork:

    def __init__(this):
        this.W = [] # list containing all the L weightmatrices
        this.b = [] # list containing all the L biasvectors
        
        # for updating iteratively
        this.W_old = [] # list containing all the L weightmatrices
        this.b_old = [] # list containing all the L biasvectors
        
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
"""
        this.delta = None # the deltas in the backpropagation
"""

    #
    # Functions for setting up the neural network
    #

    
    # Add a layer to the end of the network
    #
    # @input_dim: The number of inputs to this layer
    # @output_dim: The number of outputs from this layer
    # @act_func: The activation function of this layer
    # @act_func_diff: The derivative of the activation function 
    #                 of this layer
    #
    def add_layer(this, input_dim, output_dim, act_func, act_func_diff):
        # set up the weights of the layer
        this.W.append((random.random((output_dim, inpu_dim)) - .5)*.7/.5)
        # set up the biases of the layer
        this.b.append((random.random(output_dim) - .5)*.7/.5)
        # store the activation function of the layer
        this.act.append(act_func)
        # store the derivative of the activation function of the layer
        this.act_diff.append(act_func_diff)


    # Setting the cost function for the network
    def set_cost_function(this, cost_func, cost_func_diff):
        this.cost_func = cost_func
        this.cost_func_diff = cost_func_diff


    def set_learning_rate(this, eta):
        this.eta = eta # the learning rate of the network

    
    # This function must be run after the network is specified and before
    # training 
    def compile(this):
        print("This function is not fully implemented yet.")
        
        if (len(this.W) == 0):
            print("No layers. Compilation unsuccessfull.")
            return

        if (this.cost_func == None):
            print("Cost function is missing. Compilation unsuccessfull.")
            return

        if (this.cost_func_diff == None):
            print("Cost function derivative is missing. 
                    Compilation unsuccessfull.")
            return
        

        # set up list of vectors of deltas
        this.delta = [zeros((len(this.b[0]), 1))]
        for i in range(len(b) - 1):
            this.delta.append(zeros((len(this.b[i+1]), 1))]

        # set up list of gradientmatrices C(W)
        
        # set up list of gradientvectors C(b)

        this.compiled = True



    #
    # Functions for training and prediction 
    #


    # Predicts the response/output given a predictor/input
    #
    # @x: the input/predictor for the neural network to predict on as
    #     a column vector
    #
    def predict(this, x):
        if !this.compiled:
            print("Network is not compiled. Use compile().")
            return
        
        y = x
        
        # compute the output
        for i in range(len(this.W)):
            y = this.act_func[i](this.W[i].dot(y) + this.b[i])

        return z, y


    # Predicts the response/output given a predictor/input
    # Also saves the activations, z, in the network for the given input
    #
    # @x: the input/predictor for the neural network to predict on as
    #     a column vector
    #
    def forward(this, x):
        if !this.compiled:
            print("Network is not compiled. Use compile().")
            return
        
        # set up list of vectors of activations
        z = [zeros((len(this.b[0]), 1))]
        for i in range(len(b) - 1):
            z.append(zeros((len(this.b[i+1]), 1))]
        
        y = x

        # compute the output
        for i in range(len(this.W)):
            z[i] = this.W[i].dot(y) + this.b[i]
            y = this.act_func[i](z[i])

        return z, y
    
    
    # The much talked about backpropagation
    # 
    # @x: predictor/input
    # @z: list of the activations for each layer in the network given x as
    #     input
    # @t: targets
    #
    def back_propagate(this, x, z, t):
        print("This function is not fully implemented yet.")
        
        if !this.compiled:
            print("Network is not compiled. Use compile().")
            return
        
        # setting up neede variables, lists and so on
        L = len(this.W)
        delta = [None]*len(L)
        
        # set up list of gradients with respect to the weights and biases
        gradient_W = []
        gradient_b = []
        for i in range(L):
            gradient_W.append(zeros((this.W[i].shape))]
            gradient_b.append(zeros((this.b[i].shape))]
        

        # Compute delta_L
        delta[L] += this.cost_func_diff(this.act_func[L](z[L]), t)*this.act_func_diff[L](z[L])

        # set up "iterator"
        range_L = reversed(range(L))
        range_L.pop()

        # Compute delta_l for l=L-1, L-2, ..., 1
        for l in range_L:
            delta[l - 1] += sum(delta[l]*this.W[:,l:l+1]
                    *this.act_func_diff[l-1](z[l-1]), axis=1)
        
        # special treatment of last iteration
        delta[0] += sum(delta[1]*this.W[:,1:2]
                *x, axis=1)

        # compute the gradients
        for l in range_L:
            gradient_W[:,l] = delta[l]*this.act[l-1](z[l-1])).T # CHECK
            gradient_b[l] = delta[l]
        
        # special treatment of last iteration
        gradient_W[:,0] = delta[0]*x.T # CHECK
        gradient_b[0] = delta[0]


        return gradient_W, gradient_b


    # computes the gradient with respect to the weights and biases for a 
    # given sample x 
    def compute_gradient(this, x, t):

        z, y = forward(x)
        gradient_W, gradient_b = back_propagate(this, x, z, t)

        return gradient_W, gradient_b


    # Fit yor network using stochastic gradient method
    #
    # @X: trainingdatamatrix, columns are datapoints
    # @T: targets for training data X, columns are datapoints
    #
    def fit(this, X, T, iters=5, batch_size=10):
        print("This function is not fully implemented yet.")
        
        if !this.compiled:
            print("Network is not compiled. Use compile().")
            return

        L = len(this.w)
        
        # FIX: send the gradient list as argument to compute_gradient

        # set up temporary gradient and weight holders and next weights,grds
        gradient_W = []
        gradient_b = []
        W_next = []
        b_next = []

        for l in range(L):
            gradient_W.append(zeros((this.W[l].shape))]
            gradient_b.append(zeros((this.b[l].shape))]
            W_next.append(zeros((this.W[l].shape))]
            b_next.append(zeros((this.b[l].shape))]
        
        # learn network using stochastic gradient method 
        for i in range(iters):
            # pick random numbers from 0 to x.shape 
            batch = choice(X.shape[1], batch_size, replace=False)
            total_grad_W = 0
            total_grad_b = 0

            for j in batch:
                # compute the gradient of the costfunction for sample j 
                smpl_grad_W, smpl_grad_b = compute_gradient(X[:,j], T[:,j])

                # add this gradient/N to the total gradient
                for l in range(L):
                    gradient_W[l] += smpl_gradient_W/batch_size
                    gradient_b[l] += smpl_gradient_b/batch_size

            # update weights and biases
            for l in range(L):
                W_next[l] = this.W[l] - this.eta*gradient_W[l]
                this.W[l] = W_next[l]
                b_next[l] = this.b[l] - this.eta*gradient_b[l]
                this.b[l] = b_next[l]





