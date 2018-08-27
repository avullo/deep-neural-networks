import numpy as np

from .activation_functions import sigmoid, sigmoid_backward, relu, relu_backward 

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

class NeuralNetwork:
    def __init__(self, layer_dims, seed = 1):
        self.layer_dims = layer_dims
        self.initialize_parameters(seed)

    #
    # TODO
    # Add various initialization methods
    #
    def initialize_parameters(self, seed = 1):
        """
        Initialise network parameters

        Arguments:
          seed -- seed controlling random number generation
        
        """

        np.random.seed(seed)
        layer_dims = self.layer_dims # includes input layer
        L = len(layer_dims) - 1 # number of network layers, excluding input one
        self.L = L
        
        # the set of weights and biases and their corresponding derivatives
        # first element is not used in order to be able to index according to layer number
        self.W, self.dW = [ None ] * (L+1), [ None ] * (L+1)
        self.b, self.db = [ None ] * (L+1), [ None ] * (L+1)
        
        W, b = self.W, self.b
        for l in range(1, L+1):
            W[l] = np.random.randn(layer_dims[l], layer_dims[l-1]) * .01
            b[l] = np.zeros((layer_dims[l], 1))

            assert(W[l].shape == (layer_dims[l], layer_dims[l-1]))
            assert(b[l].shape == (layer_dims[l], 1))

        # list of caches (tuples) containing one cache of _linear_activation_forward for each of the L layers
        self.caches = []

    def L_model_forward(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
        Arguments:
          X -- data, numpy array of shape (input size, number of examples)
            
        Returns:
          AL     -- last post-activation value
        """

        self.caches = []
        A = X
        L = self.L # number of layers in the neural network
    
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            # TODO: no need to return linear cache, is just A_prev
            A, cache = self._linear_activation_forward(A_prev, l, "relu")
            self.caches.append(cache)
    
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self._linear_activation_forward(A, L, "sigmoid")
        self.caches.append(cache)
    
        assert(AL.shape == (1,X.shape[1]))
        
        return AL

    def L_model_backward(self, AL, Y):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
          AL     -- probability vector, output of the forward propagation (L_model_forward())
          Y      -- true "label" vector (containing 0 if false class, 1 otherwise)

        """
        L = self.L
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
    
        # Initializing the backpropagation
        # TODO: make it generic depending on the cost function
        dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, Lth layer number. Output: dA_L-1
        dA_prev = self._linear_activation_backward(dAL, L, activation="sigmoid")
    
        # Reverse loop through the previous layers
        for l in reversed(range(L-1)):
            # (l+1)-th layer: (RELU -> LINEAR) gradients.
            # Inputs: dAl+1, (l+1)-th layer number. Outputs: dAl
            dA = dA_prev
            dA_prev = self._linear_activation_backward(dA, l+1, activation="relu")

    # Gradient descent
    # TODO: implement other algorithms
    def update_parameters(self, alpha):
        """
        Update parameters using gradient descent
    
        Arguments:
          alpha -- learning rate, controls the rate of covergence
    
        """
    
        # Update rule for each parameter.
        for l in range(1, self.L+1):
            self.W[l] = self.W[l] - alpha * self.dW[l]
            self.b[l] = self.b[l] - alpha * self.db[l]

    def _linear_forward(self, A_prev, l):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
          A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
          l      -- number of the current layer

        Returns:
          Z     -- the input of the activation function, also called pre-activation parameter 
        """

        W, b = self.W[l], self.b[l]
        Z = np.dot(W, A_prev) + b
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    
        return Z

    def _linear_activation_forward(self, A_prev, l, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
          A_prev     -- activations from previous layer (or input data): (size of previous layer, number of examples)
          l          -- number of the current layer
          activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
          A     -- the output of the activation function, also called the post-activation value 
          cache -- a python tuple containing "linear_cache" (A_prev) and "activation_cache" (Z);
                   stored for computing the backward pass efficiently
        """

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z = self._linear_forward(A_prev, l)

        # TODO: remove cache return from activation functions, is just 'Z'
        if activation == "sigmoid":
            A = sigmoid(Z)    
        elif activation == "relu":
            A = relu(Z)
        assert (A.shape == (self.W[l].shape[0], A_prev.shape[1]))
        
        cache = (A_prev, Z)

        return A, cache
                
    def _linear_backward(self, dZ, l, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
          dZ    -- Gradient of the c
          l     -- number of the current layer
          cache -- A_prev

        Returns:
          dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        """
        A_prev = cache
        m = A_prev.shape[1]

        # compute gradient for current layer
        W, b = self.W[l], self.b[l]
        self.dW[l] = 1.0 / m * np.dot(dZ, A_prev.T)
        self.db[l] = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    
        assert (dA_prev.shape == A_prev.shape)
        assert (self.dW[l].shape == W.shape)
        assert (self.db[l].shape == b.shape)

        return dA_prev

    def _linear_activation_backward(self, dA, l, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
        Arguments:
          dA         -- post-activation gradient for current layer
          l          -- number of current layer
          activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
        Returns:
          dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
          dW      -- Gradient of the cost with respect to W (current layer l), same shape as W
          db      -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        # current layer cache, index range is 0 .. L-1
        linear_cache, activation_cache = self.caches[l-1]

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)

        dA_prev = self._linear_backward(dZ, l, linear_cache)

        return dA_prev
