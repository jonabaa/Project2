import warnings
from sklearn.neural_network import MLPClassifier
from NeuralNetwork import NeuralNetwork
from functions import *

X = np.array([[0.0, 2.0, 2.0, 70], [1.0, 1.2, 3.0, 18]])
y = np.array([0, 2])
print(X.shape)

mlp = MLPClassifier( solver              = 'sgd',      # Stochastic gradient descent.
                    activation          = 'logistic', # Skl name for sigmoid.
                    alpha               = 0.0,        # No regularization for simplicity.
                    momentum            = 0.0,        # Similarly no momentum to the sgd
                    max_iter            = 1,          # Only do one step per fit call
                    hidden_layer_sizes  = (3, 3))     # Full network is of size (3,3,3,1),

# Force sklearn to set up all the necessary matrices by fitting a data set. 
# We dont care if it converges or not, so lets ignore raised warnings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mlp.fit(X,y)

# We now make my network, and give it the same weights and biases
nn = NeuralNetwork()
nn.add_layer(3, sigmoid, sigmoid_diff)
nn.add_layer(3, sigmoid, sigmoid_diff)
nn.add_layer(1, sigmoid, sigmoid_diff)
nn.set_inputnodes(X.shape[1])
nn.set_cost_function(cost_crossentr, cost_crossentr_diff)
nn.compile()

# Copy the weights and biases from the scikit-learn network to your own.
# keep them for record keeping, so we can compute weight changes
old_weights = []
old_biases = []
print(mlp.coefs_[0].shape)
print(mlp.intercepts_[0].shape)
for i, w in enumerate(mlp.coefs_) :
    nn.W[i] = np.copy(w.T)
    old_weights.append(np.copy(w.T))
for i, b in enumerate(mlp.intercepts_) :
    nn.b[i] = np.reshape(np.copy(b.T), (len(b), 1))
    old_biases.append(np.reshape(np.copy(b.T), (len(b),1)))

# pick a point and check that the two networks give the same values
    
X      = np.reshape(X[1], (1, X.shape[1]))
target = np.reshape(y[1], (1,1))

print("We the weights initialized to the same thing, the two networks make same predictions")
print("My network predicts")
print(nn.predict_probability(X))
print("The mlp regressor predicts")
print(mlp.predict_proba(X))
print("")

########
# We take on backpropagation step in my network, by simply training with a single epoch. 

my_zs, y_prime = nn.forward(X)
my_activations = [X.T] + [nn.act[i](my_zs[i]) for i in range(len(my_zs))]
nn.set_learning_rate(1)
nn.fit_det_batch(X, target,  X.shape[0], epochs = 1)
new_weights = nn.W
new_biases = nn.b
# Now we can recover the gradients of the weights by looking at the weight differences 
my_coef_grads = [(ow - nw) for (ow,nw) in zip(old_weights, new_weights)]
my_bias_grads = [(ob - nb) for (ob, nb) in zip(old_biases, new_biases) ]

# All this is setup to call the _forward_pass and _backprop methods    
# ==========================================================================
n_samples, n_features   = X.shape
batch_size              = n_samples
hidden_layer_sizes      = mlp.hidden_layer_sizes
if not hasattr(hidden_layer_sizes, "__iter__"):
    hidden_layer_sizes = [hidden_layer_sizes]
hidden_layer_sizes = list(hidden_layer_sizes)
layer_units = ([n_features] + hidden_layer_sizes + [mlp.n_outputs_])
activations = [X]
activations.extend(np.empty((batch_size, n_fan_out)) 
                   for n_fan_out in layer_units[1:])
deltas      = [np.empty_like(a_layer) for a_layer in activations]
coef_grads  = [np.empty((n_fan_in_, n_fan_out_)) 
               for n_fan_in_, n_fan_out_ in zip(layer_units[:-1],
                                                layer_units[1:])]
intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
# ==========================================================================
    
activations                       = mlp._forward_pass(activations) 
loss, coef_grads, intercept_grads = mlp._backprop(X, target, activations, deltas, coef_grads, intercept_grads)
print()
print("Length of list of activationvectors: %s" % len(activations))
print("Length of list of my activationvectors: %s" % len(my_activations))
print("We first compare the activations, if they are equal we print of list of trues")
print([np.allclose(mya.T, a) for (mya, a) in zip(my_activations, activations)])
print("")

print("We then compare teh coeficient grads from mlp an my network.")
print("For each layer we ask Sci-kit learn if the coefs are equal, if they are we print a list of trues")
print([my_grad.shape for my_grad in my_coef_grads])
print([grad.shape for grad in coef_grads])
print()
print([np.allclose(my_grad.T, mlp_grad) for (my_grad, mlp_grad) in zip(my_coef_grads, coef_grads)])
print("Same deal for biases")
print([np.allclose(my_grad.T, mlp_grad) for (my_grad, mlp_grad) in zip(my_bias_grads, intercept_grads)])
print("")

"""
# Copy the weights and biases from the scikit-learn network to your own.
# keep them for record keeping, so we can compute weight changes
old_weights = []
old_biases = []
print(mlp.coefs_[0].shape)
print(mlp.intercepts_[0].shape)
for i, w in enumerate(mlp.coefs_) :
    nn.W[i] = np.copy(w.T)
    old_weights.append(np.copy(w.T))
for i, b in enumerate(mlp.intercepts_) :
    nn.b[i] = np.reshape(np.copy(b.T), (len(b), 1))
    old_biases.append(np.reshape(np.copy(b.T), (len(b),1)))

# pick a point and check that the two networks give the same values
    
X      = np.reshape(X[1], (1, X.shape[1]))
target = np.reshape(y[1], (1,1))
"""

print("We the weights initialized to the same thing, the two networks make same predictions")
print("My network predicts")
print(nn.predict_probability(X))
print("The mlp regressor predicts")
print(mlp.predict_proba(X))
print("")

########
# We take on backpropagation step in my network, by simply training with a single epoch

my_zs, y_prime = nn.forward(X)
my_activations = [X.T] + [nn.act[i](my_zs[i]) for i in range(len(my_zs))]
nn.set_learning_rate(1)
nn.fit_det_batch(X, target,  X.shape[0], epochs = 1)
new_weights = nn.W
new_biases = nn.b

# Now we can recover the gradients of the weights by looking at the weight differences 
my_coef_grads = [(ow - nw) for (ow,nw) in zip(old_weights, new_weights)]
my_bias_grads = [(ob - nb) for (ob, nb) in zip(old_biases, new_biases) ]

# All this is setup to call the _forward_pass and _backprop methods    
# ==========================================================================
n_samples, n_features   = X.shape
batch_size              = n_samples
hidden_layer_sizes      = mlp.hidden_layer_sizes
if not hasattr(hidden_layer_sizes, "__iter__"):
    hidden_layer_sizes = [hidden_layer_sizes]
hidden_layer_sizes = list(hidden_layer_sizes)
layer_units = ([n_features] + hidden_layer_sizes + [mlp.n_outputs_])
activations = [X]
activations.extend(np.empty((batch_size, n_fan_out)) 
                   for n_fan_out in layer_units[1:])
deltas      = [np.empty_like(a_layer) for a_layer in activations]
coef_grads  = [np.empty((n_fan_in_, n_fan_out_)) 
               for n_fan_in_, n_fan_out_ in zip(layer_units[:-1],
                                                layer_units[1:])]
intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
# ==========================================================================
    
activations                       = mlp._forward_pass(activations) 
loss, coef_grads, intercept_grads = mlp._backprop(X, target, activations, deltas, coef_grads, intercept_grads)

print()
print("Length of list of activationvectors: %s" % len(activations))
print("Length of list of my activationvectors: %s" % len(my_activations))
print("We first compare the activations, if they are equal we print of list of trues")
print([np.allclose(mya.T, a) for (mya, a) in zip(my_activations, activations)])
print("")

print("We then compare teh coeficient grads from mlp an my network.")
print("For each layer we ask Sci-kit learn if the coefs are equal, if they are we print a list of trues")
print([my_grad.shape for my_grad in my_coef_grads])
print([grad.shape for grad in coef_grads])
print()
print([np.allclose(my_grad.T, mlp_grad) for (my_grad, mlp_grad) in zip(my_coef_grads, coef_grads)])
print("Same deal for biases")
print([np.allclose(my_grad.T, mlp_grad) for (my_grad, mlp_grad) in zip(my_bias_grads, intercept_grads)])
print("")

# Now train my network and deduce the deltas
#nn.train(X, target, cost_function = 'square error', learning_rate = 0.001, epochs = 1, batch_size = 1, verbose = False)

#for i, layer in enumerate(nn.layers) :
    #print(mlp.coefs_[i] - old_weights[i])
    #assert(np.allclose(mlp.coefs_[i], old_weights[i]))
    #print(layer.weights - mlp.coefs_[i])
    #assert(np.allclose(layer.weights, mlp.coefs_[i]))
    #print(layer.bias - mlp.intercepts_[i])
    #assert(np.allclose(layer.bias, mlp.intercepts_[i]))

