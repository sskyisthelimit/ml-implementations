import unittest
import numpy as np
from SVM.SVM import SVM

from utils.utils import (assert_raise_message)


class TestSVM(unittest.TestCase):
    def setUp(self):
        self.model = SVM(margin_type='hard')
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([1, 1, -1, -1])

    def test_invalid_margin_type(self):
        assert_raise_message(
            ValueError,
            "margin_type must be 'soft' or 'hard'",
            SVM, margin_type='abc')

    def test_invalid_kernel_name(self):
        assert_raise_message(
            ValueError,
            "Invalid kernel type."
            " Supported kernels: 'linear', 'rbf', 'polynomial', 'sigmoid'",
            SVM, kernel='abc')

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.alpha)
        self.assertIsNotNone(self.model.b)
        self.assertIsNotNone(self.model.w)

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        self.assertTrue((predictions == self.y).all())

    def test_invalid_predict_X_shape(self):
        self.model.fit(self.X, self.y)
        assert_raise_message(
            ValueError,
            "provided X is invalid - features don't match",
            self.model.predict,
            np.array([[1, 2, 3], [3, 4, 5]]))


if __name__ == '__main__':
    unittest.main()
