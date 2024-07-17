import os 
import sys
import unittest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from LogisticRegression.LogisticRegression import LogisticRegression  

from utils.utils import assert_true, assert_equal
from utils.utils import assert_raise_message

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.binary_logistic = LogisticRegression(
            n_iterations=1000, learning_rate=0.1)
        
        self.multiclass_logistic = LogisticRegression(
            n_iterations=1000, learning_rate=0.1, solver='softmax')
    
        self.X_binary = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_binary = np.array([0, 0, 1, 1])
        
        self.X_multiclass = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        self.y_multiclass = np.array([0, 1, 2, 0, 1, 2])

    def test_valid_binary_fit(self):
        self.binary_logistic.fit(self.X_binary, self.y_binary)
        assert_equal(self.binary_logistic.target_type, 'binary')
        assert_equal(self.binary_logistic.solver, 'sga')
        assert_true(self.binary_logistic.weights is not None)
        
    def test_valid_multiclass_fit(self):
        self.multiclass_logistic.fit(self.X_multiclass, self.y_multiclass)
        assert_equal(self.multiclass_logistic.target_type, 'multiclass')
        assert_equal(self.multiclass_logistic.solver, 'softmax')
        assert_true(self.multiclass_logistic.weights is not None)

    def test_valid_multiclass_prediction_shape(self):
        self.multiclass_logistic.fit(self.X_multiclass, self.y_multiclass)
        y_m_pred = self.multiclass_logistic.predict(
            np.array([[1, 0], [2, 1], [4, 4], [1, 2], [1, 1], [1, 1]])
        )
        assert_equal(y_m_pred.shape, self.y_multiclass.shape)

    def test_valid_binary_prediction_shape(self):
        self.binary_logistic.fit(self.X_binary, self.y_binary)
        y_b_pred = self.binary_logistic.predict(
            np.array([[1, 0], [2, 1], [4, 4], [1, 2]])
        )
        assert_equal(y_b_pred.shape, self.y_binary.shape)

    def test_invalid_binary_y(self):
        self.invalid_y = np.array([[1, 0], [0, 1], [1, 1]])
        assert_raise_message(
            ValueError,
            "Provided array has invalid shape {0}".format(self.invalid_y.shape),
            self.binary_logistic.fit, self.X_binary,
            self.invalid_y)

    def test_invalid_multiclass_target(self):
        self.invalid_m_y = np.array([1, 0.8, 1.1, 2, 1, 2])
        assert_raise_message(
            ValueError, "Couldn't determine classification target type",
            self.multiclass_logistic.fit, self.X_multiclass,
            self.invalid_m_y)

    def test_edge_cases(self):
        # Very small dataset
        X_small = np.array([[1, 2]])
        y_small = np.array([0])
        model = LogisticRegression()
        assert_raise_message(
            ValueError, "Couldn't determine classification target type",
            model.fit, X_small, y_small)

        # Very large dataset
        X_large = np.random.rand(10000, 100)
        y_large = np.random.randint(0, 2, size=10000)
        model = LogisticRegression()
        model.fit(X_large, y_large)
        predictions = model.predict(X_large)
        assert predictions.shape == y_large.shape

if __name__ == '__main__':
    unittest.main()
