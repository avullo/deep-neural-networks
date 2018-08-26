from .core import initialize_parameters, L_model_forward, cross_entropy_cost, L_model_backward, update_parameters

def train(X, Y, layers_dims, seed = 1, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
      X              -- data, numpy array of shape (input_dim, number of examples)
      Y              -- true "label" vector (containing 1 1f true classe, 0 otherwise), of shape (1, number of examples)
      layers_dims    -- list containing the input size and each layer size, of length (number of layers + 1).
      learning_rate  -- learning rate of the gradient descent update rule
      num_iterations -- number of iterations of the optimization loop
      print_cost     -- if True, it prints the cost every 100 steps
    
    Returns:
      parameters -- parameters learnt by the model. They can then be used to predict.
      costs      -- list of cost values, one every 100 iterations
    """

    costs = []
    
    # Parameters initialization
    parameters = initialize_parameters(layers_dims, seed)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = cross_entropy_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    return parameters, costs
