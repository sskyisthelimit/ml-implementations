import numpy as np
from utils.validation import check_classification_X_y, check_array


class Node:
    def __init__(self, left_child=None, right_child=None,
                 threshold=None, feature=None, *, value=None):
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.value = value


class DecisionTree:
    def __init__(self, n_features=None, min_samples=2, max_depth=150):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        X, y = check_classification_X_y(X, y)
        self.n_features = X.shape[1] if not self.n_features else min(
            self.n_features, X.shape[1])
        
        self.ovrl_features = X.shape[1]

        self.labels, self.labels_counts = np.unique(y, return_counts=True)
        self.root = self._build_tree(X, y, 1)

    def _build_tree(self, X, y, depth):
        n_rows, n_cols = X.shape
        labels, labels_counts = np.unique(y, return_counts=True)

        if len(y) < self.min_samples or len(labels) == 1 or\
           depth >= self.max_depth:
            
            return self._leaf_label(y)
        
        feature_idxs = np.random.choice(n_cols, self.n_features, replace=False)
        best_threshold, best_col_idx = self._find_best_split(X, y,
                                                             feature_idxs)

        l_idxs, r_idxs = self._split(X[:, best_col_idx], best_threshold)

        l_child = self._build_tree(X[l_idxs, :], y[l_idxs], depth=depth+1)
        r_child = self._build_tree(X[r_idxs, :], y[r_idxs], depth=depth+1)
        
        return Node(threshold=best_threshold, feature=best_col_idx,
                    left_child=l_child, right_child=r_child)
    
    def _find_best_split(self, X, y, feature_idxs):
        best_gain = -1
        best_threshold, best_col_idx = None, None

        for feature_idx in feature_idxs:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._calc_inf_gain(X[:, feature_idx], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_col_idx = feature_idx

        return best_threshold, best_col_idx

    def _split(self, feature_col, threshold):
        l_idxs = np.argwhere(feature_col <= threshold).flatten()
        r_idxs = np.argwhere(feature_col > threshold).flatten()
        return l_idxs, r_idxs

    def _calc_inf_gain(self, feature_col, y, threshold):

        def _calc_node_entropy(y):
            labels, labels_counts = np.unique(y, return_counts=True)
            labels_prob = labels_counts / np.sum(labels_counts)
            return -np.sum(labels_prob * np.log(labels_prob))

        p_entropy = _calc_node_entropy(y)

        l_idxs, r_idxs = self._split(feature_col, threshold)

        if len(l_idxs) == 0 and len(r_idxs) == 0:
            return 0

        l_entropy = _calc_node_entropy(y[l_idxs])
        r_entropy = _calc_node_entropy(y[r_idxs])

        l_count = len(l_idxs)
        r_count = len(r_idxs)
        l_weight = l_count / (l_count + r_count)
        r_weight = r_count / (l_count + r_count)
        
        return p_entropy - l_weight * l_entropy - r_weight * r_entropy

    def _leaf_label(self, y):
        labels, labels_counts = np.unique(y, return_counts=True)
        return Node(value=labels[np.argmax(labels_counts)])

    def _prediction_traversal(self, x):
        cur_node = self.root
        while cur_node.left_child and cur_node.right_child:
            
            if x[cur_node.feature] <= cur_node.threshold:
                cur_node = cur_node.left_child
            else:
                cur_node = cur_node.right_child
        return cur_node.value

    def predict(self, X):
        X = check_array(X)
        if X.shape[1] != self.ovrl_features:
            raise ValueError("provided X is invalid - features don't match")
        
        return np.array([self._prediction_traversal(x) for x in X])
