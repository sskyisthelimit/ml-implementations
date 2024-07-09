import numpy as np
import numpy.linalg as linalg
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.validation import check_X_y, check_array
from utils.utils import preprocess_data


class LinearRegression:
    def __init__(self, method="", intercept=True,
                 n_iterations=10000, learning_rate=0.001,
                 normalize=True):
        self.weights = None
        self.method = method.lower()
        self.n_iterations = n_iterations
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.X_mean = None
        self.y_mean = None
        self.l2_norm = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        X, y, self.X_mean, self.y_mean, self.l2_norm = preprocess_data(
            X, y, intercept=self.intercept, normalize=self.normalize)
        self.__normal_equations(X, y)

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet."
                             "Call fit() before predict().")
        
        X = check_array(X, ensure_all_finite=True, ensure_min_features=1,
                        ensure_min_samples=1)

        if self.normalize:
            X = (X - self.X_mean) / self.l2_norm

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = np.dot(X_b, self.weights)

        if self.intercept:
            y_pred += self.y_mean

        return y_pred

    def __normal_equations(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        transposed_X = np.transpose(X_b)
        self.weights = np.dot(np.dot(linalg.inv(np.dot(transposed_X, X_b)),
                                     transposed_X), y)
