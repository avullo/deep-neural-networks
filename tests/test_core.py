from deepnn.core import * 

import unittest
import numpy as np

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
    
        
            
