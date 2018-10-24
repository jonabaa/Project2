# This class implements a multilayered neural network

from numpy import random, zeros, sum

class NeuralNetwork:

    def __init__(this):
        this.W = [] # list containing all the L weightmatrices
        this.b = [] # list containing all the L biasvectors
        
        # for updating iteratively
        this.W_next = [] # list containing all the L weightmatrices
        this.b_next = [] # list containing all the L biasvectors
        
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

        this.z = None # the activations across the whole network
        this.delta = None # the deltas in the backpropagation

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
        
        # set up list of vectors of activations
        this.z = [zeros((len(this.b[0]), 1))]
        for i in range(len(b) - 1):
            this.z.append(zeros((len(this.b[i+1]), 1))]

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

        return y


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
        
        y = x

        # compute the output
        for i in range(len(this.W)):
            z[i] = this.W[i].dot(y) + this.b[i]
            y = this.act_func[i](z[i])

        return y


    # The much talked about backpropagation
    # 
    # @x: predictor/input
    # @t: targets
    #
    def back_propagate(this, x, t):
        print("This function is not fully implemented yet.")
        
        if !this.compiled:
            print("Network is not compiled. Use compile().")
            return
        """
        # for increased readability
        z = this.z
        delta = this.delta
        W = this.W
        b = this.b
        L = len(W)
        N = X.shape[0]

        # compute the deltas
        for k in range(N):
            """
        # forward
        y = forward(x)

        # Compute delta_L
        delta[L] += this.cost_func_diff(this.act_func[L](z[L]))*this.act_func_diff[L](z[L])

        # Compute delta_l for l=L-1, L-2, ..., 1
        for l in reversed(range(len(this.W) - 1)):
            delta[l] += sum(delta[l+1]*this.W[:,l+1:l+2]*this.act_func_diff[l](z[l]), axis=1)

        gradient =  
            
        # Update the weights and biases
        """
        for i in delta:
            i /= N
        """
        return gradient


    # Fit yor network using stochastic gradient method
    #
    # @X: trainingdatamatrix, columns are datapoints
    # @T: targets for training data X, columns are datapoints
    #
    def fit(this, X, T, iters=5, batch_size=10):
        for i in range(iters):
            # pick random numbers from 0 to X.shape[1] - 1
            batch = some_function(X, batch_size)
            for j in batch:
                # compute the gradient of the costfunction for sample j 
                gradient = back_propagate(X[:,j], T[:,j])

                # add this gradient/N to the total gradient
                total_grad += gradient/batch_size

            # update weights and biases
            for l in range(L):
                W = W_old - this.eta


    # Step 2 of the much talked about backpropagation
    def update_parameters(this):
        print("This function is not implemented yet.")
        
        if !this.compiled:
            print("Network is not compiled. Use compile().")
            return
