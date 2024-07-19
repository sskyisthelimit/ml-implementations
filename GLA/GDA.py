import numpy as np
from utils.validation import check_classification_X_y, check_array


class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.means = None
        self.classes = None
        self.classes_count = None
        self.priors = None
        self.covariance = None
        self.cov_det = None
        self.cov_inv = None
        self.n_features = None

    def fit(self, X, y):
        X, y = check_classification_X_y(X, y)

        self.n_features = X.shape[1]

        self.classes, self.classes_count = np.unique(y, return_counts=True)
        self.priors = self.classes_count / y.size

        self.means = [np.mean(X[y == y_class], axis=0) for
                      y_class in self.classes]

        centered_data = np.vstack([(X[y == cls] - mean)
                                   for cls, mean in zip(self.classes,
                                                        self.means)])
        
        self.covariance = np.cov(centered_data, rowvar=False, bias=True)
        self.cov_det = np.linalg.det(self.covariance)
        self.cov_inv = np.linalg.inv(self.covariance)

    def predict(self, X):
        if self.means is None:
            raise ValueError(
                "Data hasn't been fitted."
                " Call fit() method on train data first!")
        
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("provided X is invalid")

        predictions = []
        
        for x in X:
            probabilities = []
            
            for cls, mean, prior in zip(self.classes, self.means, self.priors):
                x_cent = x - mean

                log_likelihood = -0.5 * (np.dot(
                    np.dot(x_cent.T, self.cov_inv), x_cent) +
                    np.log(self.cov_det) + X.shape[1] * np.log(2 * np.pi))
                
                log_probability = log_likelihood + np.log(prior)

                probabilities.append(log_probability)
            
            predictions.append(self.classes[np.argmax(probabilities)])
        
        return np.array(predictions)
    