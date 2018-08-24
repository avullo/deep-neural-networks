import numpy as np

from .activation_functions import sigmoid, relu

def initialize_parameters(layer_dims):
    """
    Arguments:
      layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
      parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1)
    """

    # TODO: initalize random seed
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * .01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
      A -- activations from previous layer (or input data): (size of previous layer, number of examples)
      W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
      b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
      Z     -- the input of the activation function, also called pre-activation parameter 
      cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
        
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
      A_prev     -- activations from previous layer (or input data): (size of previous layer, number of examples)
      W          -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
      b          -- bias vector, numpy array of shape (size of the current layer, 1)
      activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
      A     -- the output of the activation function, also called the post-activation value 
      cache -- a python dictionary containing "linear_cache" and "activation_cache";
               stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
      X          -- data, numpy array of shape (input size, number of examples)
      parameters -- output of initialize_parameters_deep()
    
    Returns:
      AL     -- last post-activation value
      caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

# TODO
# generalize to multiple classes
def cross_entropy_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
      AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
      Y  -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
      cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = -1.0 / m * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
      dZ    -- Gradient of the cost with respect to the linear output (of current layer l)
      cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
      dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
      dW      -- Gradient of the cost with respect to W (current layer l), same shape as W
      db      -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1.0 / m * np.dot(dZ, A_prev.T)
    db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
