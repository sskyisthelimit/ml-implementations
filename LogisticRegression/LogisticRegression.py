import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.validation import check_X_y, check_classification_target




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
        X, y = check_X_y(X, y)
        self.target_type = check_classification_target(y)

        self.samples_n, self.features_n = X.shape

        if self.target_type == 'binary':
            if self.solver == 'sga':
                self._sga_solver(X, y)
            elif self.solver == 'newton':
                self._newtons_solver(X, y)
        elif self.target_type == 'multiclass':
            self.solver = 'softmax'
            self._softmax_solver(X, y)
        else:
            raise ValueError('Invalid target type')
        
        return self
    
    def _calc_hypothesis(self, theta, X):
        return 1 / (1 + np.exp(-np.dot(X, theta)))

    def _sga_solver(self, X, y):
        # mb check for fit or smt
        self.weights = np.full(self.features_n, self.intercept)

        for _ in range(self.n_iterations):
            hypothesis = self._calc_hypothesis(theta=self.weights, X=X)
            gradient = np.dot((y - hypothesis), X)
            self.weights += (1 / self.samples_n) * \
                self.learning_rate * gradient
            
    def _newtons_solver(self, X, y):
        self.weights = np.full(self.features_n, self.intercept)
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
        
        probabilities = [count / self.samples_n for count in counts]

        probabilities = probabilities[:-1]
        
        self.weights = np.full((self.features_n, self.classes_n),
                               self.intercept)

        for _ in range(self.n_iterations):
            probabilities = self._softmax(np.dot(X, self.weights))
            gradients = -1 / self.samples_n * \
                np.dot(X.T, (one_hot - probabilities))

            self.weights -= self.learning_rate * gradients

            # Check for convergence
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
        return expected_z / np.sum(z, axis=1, keepdims=True)

    def _one_hot(self, y):
        one_hot = np.zeros(y.shape[0], self.classes_n)
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot


