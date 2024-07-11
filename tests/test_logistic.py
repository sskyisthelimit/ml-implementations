import unittest
import numpy as np
from LogisticRegression import LogisticRegression  


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.lr_binary = LogisticRegression(
            n_iterations=1000, solver='sga', learning_rate=0.01)
        self.lr_multiclass = LogisticRegression(
            n_iterations=1000, solver='softmax', learning_rate=0.01)
        
        self.X_binary = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_binary = np.array([0, 0, 1, 1])
        
        self.X_multiclass = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        self.y_multiclass = np.array([0, 1, 2, 0, 1, 2])
    
    def test_initialization(self):
        lr = LogisticRegression()
        self.assertEqual(lr.n_iterations, 100)
        self.assertEqual(lr.intercept, 0.00)
        self.assertEqual(lr.solver, 'sga')
        self.assertEqual(lr.learning_rate, 0.001)
        self.assertIsNone(lr.weights)
        self.assertEqual(lr.threshold, 0)
    
    def test_fit_binary(self):
        self.lr_binary.fit(self.X_binary, self.y_binary)
        self.assertIsNotNone(self.lr_binary.weights)
        self.assertEqual(self.lr_binary.target_type, 'binary')
    
    def test_fit_multiclass(self):
        self.lr_multiclass.fit(self.X_multiclass, self.y_multiclass)
        self.assertIsNotNone(self.lr_multiclass.weights)
        self.assertEqual(self.lr_multiclass.target_type, 'multiclass')
        self.assertEqual(self.lr_multiclass.solver, 'softmax')
    
    def test_predict_binary(self):
        self.lr_binary.fit(self.X_binary, self.y_binary)
        predictions = self.lr_binary.predict(self.X_binary)
        self.assertEqual(predictions.shape, self.y_binary.shape)
    
    def test_predict_multiclass(self):
        self.lr_multiclass.fit(self.X_multiclass, self.y_multiclass)
        predictions = self.lr_multiclass.predict(self.X_multiclass)
        self.assertEqual(predictions.shape, self.y_multiclass.shape)
    
    def test_calc_hypothesis(self):
        theta = np.array([1, 1])
        X = np.array([[1, 2], [3, 4]])
        expected_output = 1 / (1 + np.exp(-np.dot(X, theta)))
        output = self.lr_binary._calc_hypothesis(theta, X)
        np.testing.assert_array_almost_equal(output, expected_output)
    
    def test_softmax(self):
        z = np.array([[1, 2, 3], [1, 2, 3]])
        expected_output = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        output = self.lr_multiclass._softmax(z)
        np.testing.assert_array_almost_equal(output, expected_output)
    
    def test_one_hot(self):
        y = np.array([0, 1, 2, 1])
        expected_output = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
        output = self.lr_multiclass._one_hot(y)
        np.testing.assert_array_equal(output, expected_output)


if __name__ == '__main__':
    unittest.main()
