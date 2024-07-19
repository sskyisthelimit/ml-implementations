import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from cvxopt import matrix, solvers

from utils.validation import check_classification_X_y, check_array

# TODO: multiclass


class SVM:
    def __init__(self, margin_type='soft', kernel='linear', gamma=1,
                 k=1, degree=2, max_iters=5, coef0=0, C=1.0, tol=1e-3):
        if margin_type not in ['soft', 'hard']:
            raise ValueError("margin_type must be 'soft' or 'hard'")
        if kernel not in ['linear', 'rbf', 'polynomial', 'sigmoid']:
            raise ValueError(
                "Invalid kernel type." 
                " Supported kernels: 'linear', 'rbf', 'polynomial', 'sigmoid'")
        
        self.kernel = kernel
        self.gamma = gamma
        self.margin_type = margin_type
        self.k = k
        self.K = None
        self.degree = degree    
        self.C = C
        self.tol = tol
        self.max_iters = max_iters
        self.coef0 = coef0
        self.alpha = None
        self.b = 0
        self.w = None

        self.n_samples = None
        self.n_features = None

    def run_kernel(self, X, Y=None):
        if Y is None:
            Y = X

        if self.kernel == 'linear':
            return np.dot(X, Y.T)
        elif self.kernel == 'rbf':
            return self.RBF_kernel(X, Y)
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel(X, Y)
        elif self.kernel == "sigmoid":
            return self.sigmoid_kernel(X, Y)
        else:
            raise ValueError(
                "Invalid kernel provided '%s' pick one from allowed.")    

    def RBF_kernel(self, X, Y=None):
        if Y is None:
            Y = X

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        return np.exp(-self.gamma * euclidean_distances(X, Y, squared=True))
    
    def polynomial_kernel(self, X, Y=None):
        if Y is None:
            Y = X

        return np.power((np.dot(X, Y.T) + self.k), self.degree)
        
    def sigmoid_kernel(self, X, Y=None):
        if Y is None:
            Y = X

        return np.tanh(self.gamma * np.dot(X, Y.T) + self.k)

    def solve_hard_margin(self, X, y):
        P = matrix(np.outer(y, y) * self.K, tc='d')
        q = matrix(-np.ones(self.n_samples), tc='d')
        G = matrix(-np.eye(self.n_samples), tc='d')
        h = matrix(np.zeros(self.n_samples), tc='d')
        A = matrix(y, (1, self.n_samples), 'd')
        b = matrix(0.0, tc='d')
        
        solution = solvers.qp(P, q, G, h, A, b)
        
        alpha = np.ravel(solution['x'])
        
        sv = alpha > 1e-5
        ind = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        
        self.b = np.mean([y[i] - np.sum(
            self.alpha * self.sv_y *
            self.K[ind[i], sv]) for i in range(len(self.alpha))])
        
        self.w = np.dot(self.alpha * self.sv_y, self.sv)

    def solve_soft_margin(self, X, y):
        P = matrix(np.outer(y, y) * self.K, tc='d')
        q = matrix(-np.ones(self.n_samples), tc='d')
        
        G_std = np.diag(np.ones(self.n_samples) * -1)
        G_slack = np.identity(self.n_samples)
        G = matrix(np.vstack((G_std, G_slack)), tc='d')
        
        h_std = np.zeros(self.n_samples)
        h_slack = np.ones(self.n_samples) * self.C
        h = matrix(np.hstack((h_std, h_slack)), tc='d')
        
        A = matrix(y, (1, self.n_samples), 'd')
        b = matrix(0.0, tc='d')
        
        solution = solvers.qp(P, q, G, h, A, b)
        
        alpha = np.ravel(solution['x'])
        
        sv = alpha > 1e-5
        ind = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        
        self.b = np.mean([y[i] - np.sum(
            self.alpha * self.sv_y * self.K[ind[i], sv]) for i in range(
                len(self.alpha))])
        
        self.w = np.dot(self.alpha * self.sv_y, self.sv)

    def predict(self, X):
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("provided X is invalid - features don't match")
        
        return np.sign(np.dot(X, self.w) + self.b)     

    def fit(self, X, y):
        X, y = check_classification_X_y(X, y)
        self.n_samples, self.n_features = X.shape
        self.K = self.run_kernel(X)
        
        if self.margin_type == 'hard':
            self.solve_hard_margin(X, y)
        elif self.margin_type == 'soft':
            self.solve_soft_margin(X, y)
        else:
            raise ValueError(
                "Unknown margin type '%s' pick 'soft' or 'hard' margin" %
                self.margin_type)   
