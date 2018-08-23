from deepnn.core import initialize_parameters

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
        
