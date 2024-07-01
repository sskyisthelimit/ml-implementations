import numpy as np
import numpy.linalg as linal


# TODO: VALIDATION, BGD, SGD
class LinearRegression:
    def __init__(self, method="", n_iterations=10000, learning_rate=0.001):
        self.weights = None
        self.method = method.lower()
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        if self.method == "normal_equations":
            self.__normal_equations(X=X, y=y)
        # elif self.method == "bgd":
        #     self.__BGD(X=X, y=y)
        # elif self.method == "sgd":
        #     self.__SGD(X=X, y=y)
        elif self.method.strip() == "":
            self.__normal_equations(X=X, y=y)
        else:
            raise ValueError("Method for minimization of cost function must be"
                             "one of: 'normal_equations' 'bgd' 'sgd'")

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet."
                             "Call fit() before predict().")

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return np.dot(X_b, self.weights)

    def __normal_equations(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        transposed_X = np.transpose(X_b)
        self.weights = np.dot(np.dot(linal.inv(np.dot(transposed_X, X_b)),
                                     transposed_X), y)

    # def __BGD(self, X, y):

    #     n_data_points, n_features = np.shape(X)
    #     X_b = np.c_[np.ones((n_data_points, 1)), X]
    #     self.weights = np.zeros(n_features + 1)

    #     for _ in range(self.n_iterations):

    #         y_pred = np.dot(X_b, self.weights)
    #         gradient = np.dot(np.transpose(X_b), y - y_pred) / n_data_points
    #         self.weights = self.weights + self.learning_rate * gradient

    #         if (np.any(np.isnan(self.weights)) or
    #            np.any(np.isinf(self.weights))):

    #             raise ValueError("Numerical instability"
    #                              "detected during BGD.")

    # def __SGD(self, X, y):

    #     n_data_points, n_features = np.shape(X)
    #     X_b = np.c_[np.ones((n_data_points, 1)), X]
    #     self.weights = np.zeros(n_features + 1)

    #     for c in range(self.n_iterations):

    #         for i in range(n_data_points):

    #             y_pred = np.dot(X_b[i], self.weights)
    #             error = y[i] - y_pred
    #             gradient = X_b[i] * error
    #             self.weights += self.learning_rate * gradient

    #             if (np.any(np.isnan(self.weights)) or
    #                np.any(np.isinf(self.weights))):

    #                 raise ValueError("Numerical instability"
    #                                  "detected during SGD.")
