import unittest
import numpy as np

from utils.utils import assert_true, assert_equal
from utils.utils import assert_raise_message

from Tree.DecisionTree import DecisionTree, Node


class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        self.tree = DecisionTree()

    def test_init(self):
        assert_equal(self.tree.min_samples, 2)
        assert_equal(self.tree.max_depth, 150)
        assert_equal(self.tree.n_features, None)
        assert_equal(self.tree.root, None)

    def test_build_tree_stop_conditions(self):
        X = np.array([[2, 3, 3, 4, 5], [5, 4, 5, 2, 1], [3, 4, 5, 7, 7],
                      [1, 1, 2, 5, 6], [1, 1, 1, 8, 8]])
        y = np.array([1, 0, 1, 2, 2])
        self.tree.min_samples = 6
        self.tree.max_depth = 1
        res = self.tree._build_tree(X, y, 1) 
        assert_true(isinstance(res, Node))
        assert_true(res.value is not None)

    def test_split(self):
        X = np.array([1, 2, 6, 8])
        threshold = 3
        l_idxs, r_idxs = self.tree._split(X, threshold)
        np.testing.assert_array_equal(l_idxs, [0, 1])
        np.testing.assert_array_equal(r_idxs, [2, 3])

    def test_calc_inf_gain(self):
        X = np.array([1, 3, 5, 7])
        y = np.array([0, 0, 1, 1])
        threshold = 4
        gain = self.tree._calc_inf_gain(X, y, threshold)
        self.assertGreaterEqual(gain, 0)

    def test_leaf_label(self):
        y = np.array([0, 0, 1, 1, 1])
        leaf = self.tree._leaf_label(y)
        self.assertEqual(leaf.value, 1)

    def test_prediction_traversal(self):
        node1 = Node(value=1)
        node0 = Node(value=0)
        root = Node(left_child=node0, right_child=node1, feature=0, threshold=2)
        self.tree.root = root
        self.assertEqual(self.tree._prediction_traversal([1]), 0)
        self.assertEqual(self.tree._prediction_traversal([3]), 1)

    def test_predict(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 0, 1, 1])
        self.tree.fit(X_train, y_train)
        X_test = np.array([[2, 3], [6, 7]])
        predictions = self.tree.predict(X_test)
        self.assertEqual(len(predictions), 2)

if __name__ == '__main__':
    unittest.main()
