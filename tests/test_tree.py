import unittest
import numpy as np

from utils.utils import assert_true, assert_equal

from Tree.DecisionTree import DecisionTree, Node
from Tree.RandomForest import RandomForest


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
        root = Node(left_child=node0, right_child=node1,
                    feature=0, threshold=2)
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


class TestRandomForest(unittest.TestCase):
    
    def setUp(self):
        self.min_samples = 2
        self.n_trees = 6
        self.n_features = 10
        self.max_depth = 20
        self.forest = RandomForest(n_trees=self.n_trees,
                                   n_features=self.n_features, 
                                   min_samples=self.min_samples,
                                   max_depth=self.max_depth)
        self.valid_X = np.array([
            [2, 3, 3, 4, 5], [5, 4, 5, 2, 1], [3, 4, 5, 7, 7],
            [1, 1, 2, 5, 6], [1, 1, 1, 8, 8]])
        
        self.valid_y = np.array([1, 0, 1, 2, 2])

        self.bin_X = np.array([[1, 3], [5, 7], [1, 1], [2, 5]])
        self.bin_y = np.array([0, 0, 1, 1])
        
    def test_valid_trees_params(self):
        self.forest.fit(self.valid_X, self.valid_y)
        assert_equal(self.forest.n_features, 5)
        assert_equal(self.forest.n_samples, 5)
        assert_equal(self.forest.ovrl_features, 5)
        assert_equal(self.n_trees, len(self.forest.trees))
        for tree in self.forest.trees:
            assert_equal(tree.n_features, self.forest.n_features)
            assert_equal(self.min_samples, tree.min_samples)
            assert_equal(self.max_depth, tree.max_depth)

    def test_bootstrap_samples(self):
        self.forest.fit(self.valid_X, self.valid_y)
        b_X, b_y = self.forest._bootstrap_samples(self.valid_X, self.valid_y)

        assert_equal(len(self.valid_X), len(b_X))
        assert_equal(len(self.valid_y), len(b_y))

    def test_most_common_label(self):
        # first most common -  1 , second - 2
        most_common = self.forest._most_common_label(self.valid_y)
        assert_true(most_common == 1)

    def test_predict_invalid_shape(self):
        rf = RandomForest(n_trees=3)
        rf.fit(self.valid_X, self.valid_y)
        X_invalid = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        with self.assertRaises(ValueError):
            rf.predict(X_invalid)

    def test_predict(self):
        rf = RandomForest(n_trees=3)
        rf.fit(self.bin_X, self.bin_y)
        predictions = rf.predict(self.bin_X)
        self.assertEqual(predictions.shape, self.bin_y.shape)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_predict_multiclass(self):
        self.forest.fit(self.valid_X, self.valid_y)
        predictions = self.forest.predict(self.valid_X)
        self.assertEqual(predictions.shape, self.valid_y.shape)
        self.assertTrue(np.all(np.isin(predictions, [0, 1, 2])))


if __name__ == '__main__':
    unittest.main()
