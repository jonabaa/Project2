##################################
# BIN FOR OLD STUFF I MIGHT NEED #
##################################
#
#
#
#

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
                
                # SET GRADIENT TO ZERO
                gradient_W = [zeros(this.W[l].shape) for l in range(L)]
                gradient_b = [zeros(this.b[l].shape) for l in range(L)]

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
            
            # SET GRADIENT TO ZERO
            gradient_W = [zeros(this.W[l].shape) for l in range(L)]
            gradient_b = [zeros(this.b[l].shape) for l in range(L)]
            
            # compute the increase in cost after this iteration
            # (we prefer this to be a negative number then)
            cost += sum(this.cost_func(this.predict_probability(X[batches[i],:]), T[batches[i],:]))
            print("Cost increment: %f" % cost)
        


