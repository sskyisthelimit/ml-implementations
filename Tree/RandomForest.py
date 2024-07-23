import numpy as np
from collections import Counter 
from Tree.DecisionTree import DecisionTree
from utils.validation import check_classification_X_y, check_array 


class RandomForest:
    def __init__(self, n_trees=8, n_features=None,
                 min_samples=2, max_depth=50):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        X, y = check_classification_X_y(X, y)
        self.n_features = X.shape[1] if not self.n_features else min(
            self.n_features, X.shape[1])
        self.n_samples, self.ovrl_features = X.shape

        for idx in range(self.n_trees):
            tree = DecisionTree(n_features=self.n_features,
                                min_samples=self.min_samples,
                                max_depth=self.max_depth)
            
            X_samples, y_samples = self._bootstrap_samples(X, y)
            tree.fit(X_samples, y_samples)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        idxs = np.random.choice(self.n_samples, self.n_samples, replace=True)    
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        X = check_array(X)
        if X.shape[1] != self.ovrl_features:
            raise ValueError("provided X is invalid - features don't match")
        
        predictions = np.array([self.trees[i].predict(X)
                                for i in range(self.n_trees)])
        
        predictions = np.swapaxes(predictions, 0, 1)
        
        return np.array([self._most_common_label(pred)
                         for pred in predictions])