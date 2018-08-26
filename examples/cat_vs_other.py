# to be used if run from its directory
# import sys
# sys.path.insert(0, '..') 

import os
import numpy as np
import h5py

from deepnn.model import L_model_forward, train #, test

def load_data():
    # data sets size is (m, num_px, num_px, 3) where 3 is for the 3 RGB channels
    train_set = h5py.File(os.path.dirname(os.path.realpath(__file__)) + '/data/cat_vs_other.train.h5', "r")
    train_x_orig = np.array(train_set["train_set_x"][:]) # train set features
    train_y_orig = np.array(train_set["train_set_y"][:]) # train set labels

    test_set = h5py.File(os.path.dirname(os.path.realpath(__file__)) + '/data/cat_vs_other.test.h5', "r")
    test_x_orig = np.array(test_set["test_set_x"][:]) # test set features
    test_y_orig = np.array(test_set["test_set_y"][:]) # test set labels
    
    # reshape the features
    # size is (m, num_px, num_px, 3) where 3 is for the 3 RGB channels
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T # "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    assert( train_x.shape == (12288, 209) )
    assert( test_x.shape  == (12288, 50) )

    # reshape labels
    train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y  = test_y_orig.reshape((1, test_y_orig.shape[0]))
    assert( train_y.shape == (1, 209) )
    assert( test_y.shape  == (1, 50) )
    
    return ( { 'x': train_x, 'y': train_y }, { 'x': test_x, 'y': test_y } )

def predict(X, parameters):
    """
    Predict the results of a  L-layer neural network.
    
    Arguments:
      X -- data set of examples you would like to label
      parameters -- parameters of the trained model
    
    Returns:
      outputs -- predictions for the given dataset X
    """

    # forward propagation
    outputs, caches = L_model_forward(X, parameters)

    return outputs

def accuracy(probs, y):
    assert(probs.shape == y.shape)
    assert(probs.shape[0] == 1)
    m = probs.shape[1]

    # convert probabilities to 0/1 predictions
    p = np.zeros((1, m))
    for i in range(0, m):
        p[0,i] = 1 if probs[0,i] > 0.5 else 0

    return np.sum((p == y) / m)
                  

if __name__ == '__main__':
    (train_set, test_set) = load_data()

    #
    # train a 4-layer model
    # - no regularisation
    # - no hyper-parameters search
    #
    layers_dims = [ train_set["x"].shape[0], 20, 7, 5, 1 ]
    parameters, costs = train(train_set["x"], train_set["y"], layers_dims, num_iterations = 2500, print_cost = True)

    #
    # compute accuracy on the training/test set
    #
    # binary classification with sigmoid: outputs are probabilities
    train_probs = predict(train_set["x"], parameters)
    print("Accuracy: {0:.2f} (training set)".format(accuracy(train_probs, train_set["y"])))

    test_probs  = predict(test_set["x"], parameters)
    print("Accuracy: {0:.2f} (test set)".format(accuracy(test_probs, test_set["y"])))
    
    
