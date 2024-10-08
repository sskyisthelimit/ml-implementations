import numpy as np

from utils.validation import check_classification_X_y


class LogisticRegression:
    def __init__(self, n_iterations=100, intercept=0.00, solver='sga',
                 learning_rate=0.001):
        self.n_iterations = n_iterations
        self.intercept = intercept
        self.weights = None
        self.threshold = 0
        self.solver = solver
        self.samples_n = None
        self.features_n = None
        self.learning_rate = learning_rate
        self.target_type = None
        self.tol = 1e-4
        self.classes_n = None

    def fit(self, X, y):
        X, y, self.target_type = check_classification_X_y(
            X, y, return_target=True)
        
        self.samples_n, self.features_n = X.shape

        if self.target_type == 'binary':
            if self.solver in ['sga', 'newton']:
                self._binary_fit(X, y)
            else:
                raise ValueError(
                    "Invalid solver %s for binary target type " %
                    self.solver)
        elif self.target_type == 'multiclass':
            if self.solver == 'softmax':
                self._multiclass_fit(X, y)
            else:
                raise ValueError(
                    "Invalid solver %s for multiclass target type " %
                    self.solver)
        else:
            raise ValueError(
                'Invalid target type.'
                ' Only binary and discrete multiclass allowed!'
                ' Please recheck y')
        return self

    def _binary_fit(self, X, y):
        if self.solver == 'sga':
            self._sga_solver(X, y)
        elif self.solver == 'newton':
            self._newtons_solver(X, y)
        else:
            raise ValueError(
                "Invalid solver %s for binary target type " % self.solver)

    def _multiclass_fit(self, X, y):
        self._softmax_solver(X, y)
    
    def _calc_hypothesis(self, theta, X):
        return 1 / (1 + np.exp(-np.dot(X, theta)))

    def _sga_solver(self, X, y):
        self.weights = np.full(self.features_n, self.intercept)

        for _ in range(self.n_iterations):
            hypothesis = self._calc_hypothesis(theta=self.weights, X=X)
            gradient = np.dot(X.T, (y - hypothesis))
            self.weights += (1 / self.samples_n) * \
                self.learning_rate * gradient
            
    def _newtons_solver(self, X, y):
        self.weights = np.full(self.features_n, self.intercept)
        for _ in range(self.n_iterations):
            hypothesis = self._calc_hypothesis(theta=self.weights, X=X)
            M = hypothesis * (1 - hypothesis)
            gradient = (1 / self.samples_n) * np.dot(X.T, (hypothesis - y))
            hessian = (1 / self.samples_n) * np.dot(X.T * M, X)
            hessian_inv = np.linalg.inv(hessian)
            self.weights -= np.dot(hessian_inv, gradient)
     
    def _calc_log_likelihood(self, X, y):
        self.weights = np.full(self.features_n, self.intercept)
        hypothesis = self._calc_hypothesis(self.weights, X)

        for _ in range(self.n_iterations):
            self.weights -= np.dot((y - hypothesis), X)

    def _softmax_solver(self, X, y):
        possible_outcomes, counts = np.unique(y, return_counts=True)
        self.classes_n = len(possible_outcomes)
        one_hot = self._one_hot(y)
        
        self.weights = np.full((self.features_n, self.classes_n),
                               self.intercept)

        for _ in range(self.n_iterations):
            probabilities = self._softmax(np.dot(X, self.weights))
            gradients = -1 / self.samples_n * \
                np.dot(X.T, (one_hot - probabilities))

            self.weights -= self.learning_rate * gradients

            if np.linalg.norm(gradients, ord=1) < self.tol:
                break

    def predict(self, X):
        if self.solver == 'softmax':
            scores = np.dot(X, self.weights)
            probabilities = self._softmax(scores)
            return np.argmax(probabilities, axis=1)
        elif self.solver == 'sga' or self.solver == 'newton':
            scores = self._calc_hypothesis(self.weights, X)
            return (scores >= 0.5).astype(int)
    
    def _softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        expected_z = np.exp(z)
        return expected_z / np.sum(expected_z, axis=1, keepdims=True)

    def _one_hot(self, y):
        one_hot = np.zeros((y.shape[0], self.classes_n))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot
