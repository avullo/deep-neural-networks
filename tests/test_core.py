from deepnn.core import *

import unittest
import numpy as np
import pprint

class TestCore(unittest.TestCase):
    def test_initialization(self):
        np.random.seed(3)

        parameters = initialize_parameters([5,4,3])
        try:
            np.testing.assert_allclose(parameters["W1"],
                                       [[ 0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
                                        [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
                                        [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
                                        [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed initialize_parameters W1")
            
        try:
            np.testing.assert_array_equal(parameters["b1"], [[0.], [0.], [0.], [0.]])
        except AssertionError:
            self.fail("Failed initialize_parameters b1")
            
        try:
            np.testing.assert_allclose(parameters["W2"],
                                       [[-0.01185047, -0.0020565, 0.01486148, 0.00236716],
                                        [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
                                        [-0.00768836, -0.00230031, 0.00745056, 0.01976111]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed initialize_parameters W2")

        try:
            np.testing.assert_array_equal(parameters["b2"], [[0.], [0.], [0.]])
        except AssertionError:
            self.fail("Failed initialize_parameters b2")

    def test_linear_forward(self):
        np.random.seed(1)
        # TODO
        # make this general, shared with the following test case
        A_prev = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        ###
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        try:
            np.testing.assert_allclose(Z, [[ 3.26295337, -1.23429987]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed linear_forward")

    def test_linear_activation_forward(self):
        np.random.seed(2)
        A_prev = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)

        A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
        try:
            np.testing.assert_allclose(A, [[ 0.96890023, 0.11013289 ]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed linear_activation_forward with sigmoid")

        A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
        try:
            np.testing.assert_allclose(A, [[ 3.43896131, 0. ]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed linear_activation_forward with ReLU")

    def test_L_model_forward(self):
        np.random.seed(6)
        X = np.random.randn(5,4)
        W1 = np.random.randn(4,5)
        b1 = np.random.randn(4,1)
        W2 = np.random.randn(3,4)
        b2 = np.random.randn(3,1)
        W3 = np.random.randn(1,3)
        b3 = np.random.randn(1,1)
  
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}

        AL, caches = L_model_forward(X, parameters)
        try:
            np.testing.assert_allclose(AL, [[ 0.03921668, 0.70498921, 0.19734387, 0.04728177]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed L_model_forward with AL")

        self.assertEqual(len(caches), 3)
    
    def test_cross_entropy(self):
        Y = np.asarray([[1, 1, 1]])
        aL = np.array([[.8,.9,0.4]])

        self.assertTrue(abs(cross_entropy_cost(aL, Y) - 0.41493159961539694) <= 1e-3)

    def test_linear_backward(self):
        np.random.seed(1)
        dZ = np.random.randn(1,2)
        A_prev = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        linear_cache = (A_prev, W, b)

        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        try:
            np.testing.assert_allclose(dA_prev, [[ 0.51822968, -0.19517421], [-0.40506361, 0.15255393], [ 2.37496825, -0.89445391]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed linear_backward with dA_prev")

        try:
            np.testing.assert_allclose(dW, [[-0.10076895, 1.40685096, 1.64992505]] , rtol=1e-5)
        except AssertionError:
            self.fail("Failed linear_backward with dW")

        try:
            np.testing.assert_allclose(db, [[ 0.50629448]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed linear_backward with db")

    def test_linear_activation_backward(self):
        np.random.seed(2)
        dA = np.random.randn(1,2)
        A_prev = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        Z = np.random.randn(1,2)
        linear_cache = (A_prev, W, b)
        activation_cache = Z
        linear_activation_cache = (linear_cache, activation_cache)

        dA_prev, dW, db = linear_activation_backward(dA, linear_activation_cache, activation = "sigmoid")
        try:
            np.testing.assert_allclose(dA_prev, [[ 0.11017994, 0.01105339], [ 0.09466817, 0.00949723], [-0.05743092, -0.00576154]], rtol=1e-5)
            np.testing.assert_allclose(dW, [[ 0.10266786, 0.09778551, -0.01968084]], rtol=1e-5)
            np.testing.assert_allclose(db, [[-0.05729622]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed linear_activation_backward with sigmoid")

        dA_prev, dW, db = linear_activation_backward(dA, linear_activation_cache, activation = "relu")
        try:
            np.testing.assert_allclose(dA_prev, [[ 0.44090989, 0. ], [ 0.37883606, 0. ], [-0.2298228, 0. ]], rtol=1e-5)
            np.testing.assert_allclose(dW, [[ 0.44513824, 0.37371418, -0.10478989]], rtol=1e-5)
            np.testing.assert_allclose(db, [[-0.20837892]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed linear_activation_backward with ReLU")

    def test_L_model_backward(self):
        np.random.seed(3)
        AL = np.random.randn(1, 2)
        Y = np.array([[1, 0]])

        A0 = np.random.randn(4,2)
        W1 = np.random.randn(3,4)
        b1 = np.random.randn(3,1)
        Z1 = np.random.randn(3,2)
        linear_activation_cache_1 = ((A0, W1, b1), Z1)

        A1 = np.random.randn(3,2)
        W2 = np.random.randn(1,3)
        b2 = np.random.randn(1,1)
        Z2 = np.random.randn(1,2)
        linear_activation_cache_2 = ((A1, W2, b2), Z2)

        caches = (linear_activation_cache_1, linear_activation_cache_2)

        grads = L_model_backward(AL, Y, caches)
        try:
            np.testing.assert_allclose(grads["dA1"], [[ 0.12913162, -0.44014127], [-0.14175655, 0.48317296], [ 0.01663708, -0.05670698]], rtol=1e-5)
            np.testing.assert_allclose(grads["dW1"], [[ 0.41010002, 0.07807203, 0.13798444, 0.10502167],
                                                      [ 0., 0., 0., 0. ],
                                                      [ 0.05283652, 0.01005865, 0.01777766, 0.0135308 ]], rtol=1e-5)
            np.testing.assert_allclose(grads["db1"], [[-0.22007063], [ 0. ], [-0.02835349]] , rtol=1e-5)
        except AssertionError:
            self.fail("Failed L_model_backward")

    def test_update_parameters(self):
        np.random.seed(2)
        W1 = np.random.randn(3,4)
        b1 = np.random.randn(3,1)
        W2 = np.random.randn(1,3)
        b2 = np.random.randn(1,1)
        parameters = { "W1": W1, "b1": b1, "W2": W2, "b2": b2 }
        
        np.random.seed(3)
        dW1 = np.random.randn(3,4)
        db1 = np.random.randn(3,1)
        dW2 = np.random.randn(1,3)
        db2 = np.random.randn(1,1)
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2 }

        parameters = update_parameters(parameters, grads, 0.1)
        try:
            np.testing.assert_allclose(parameters["W1"], [[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
                                                          [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
                                                          [-1.0535704, -0.86128581, 0.68284052, 2.20374577]], rtol=1e-5)
            np.testing.assert_allclose(parameters["b1"], [[-0.04659241], [-1.28888275], [ 0.53405496]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed update_parameters in first layer")

        try:
            np.testing.assert_allclose(parameters["W2"], [[-0.55569196, 0.0354055, 1.32964895]], rtol=1e-5)
            np.testing.assert_allclose(parameters["b2"], [[-0.84610769]], rtol=1e-5)
        except AssertionError:
            self.fail("Failed update_parameters in second layer")
