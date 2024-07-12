
import numpy as np

class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.means = None
        self.classes = None
        self.classes_count = None
        self.priors = None
        self.covariance = None
        self.cov_det = None
        self.cov_inv = None

    def fit(self, X, y):
        self.classes, self.classes_count = np.unique(y, return_counts=True)
        self.priors = self.classes_count / y.size

        self.means = [np.mean(X[y == y_class], axis=0) for y_class in self.classes]

        centered_data = np.vstack([(X[y == cls] - mean)
                                   for cls, mean in zip(self.classes, self.means)])
        
        self.covariance = np.cov(centered_data, rowvar=False, bias=True)
        self.cov_det = np.linalg.det(self.covariance)
        self.cov_inv = np.linalg.inv(self.covariance)

    def predict(self, X):
        predictions = []
        
        for x in X:
            probabilities = []
            
            for cls, mean, prior in zip(self.classes, self.means, self.priors):
                x_cent = x - mean

                log_likelihood = -0.5 * (np.dot(np.dot(x_cent.T, self.cov_inv),
                    x_cent) + np.log(self.cov_det) + X.shape[1] * np.log(2 * np.pi))
                
                log_probability = log_likelihood + np.log(prior)

                probabilities.append(log_probability)
            
            predictions.append(self.classes[np.argmax(probabilities)])
        
        return np.array(predictions)