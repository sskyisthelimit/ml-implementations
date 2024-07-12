import unittest
import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from GLA.GDA import GaussianDiscriminantAnalysis


class TestGaussianDiscriminantAnalysis(unittest.TestCase):
    def setUp(self):
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.X_train, self.X_test,
        self.y_train, self.y_test = train_test_split(self.X, self.y,
                                    test_size=0.2, random_state=21)
        self.gda = GaussianDiscriminantAnalysis()

    def test_initialization(self):
        self.assertIsNone(self.gda.means)
        self.assertIsNone(self.gda.classes)
        self.assertIsNone(self.gda.classes_count)
        self.assertIsNone(self.gda.priors)
        self.assertIsNone(self.gda.covariance)
        self.assertIsNone(self.gda.cov_det)
        self.assertIsNone(self.gda.cov_inv)

    def test_fit(self):
        self.gda.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.gda.means)
        self.assertIsNotNone(self.gda.classes)
        self.assertIsNotNone(self.gda.classes_count)
        self.assertIsNotNone(self.gda.priors)
        self.assertIsNotNone(self.gda.covariance)
        self.assertIsNotNone(self.gda.cov_det)
        self.assertIsNotNone(self.gda.cov_inv)
        self.assertEqual(len(self.gda.means), len(np.unique(self.y_train)))

    def test_predict(self):
        self.gda.fit(self.X_train, self.y_train)
        y_pred = self.gda.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreater(accuracy, 0.5)

    def test_integration(self):
        self.gda.fit(self.X_train, self.y_train)
        y_pred = self.gda.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'GDA Accuracy: {accuracy}')
        self.assertGreater(accuracy, 0.7)

    def test_edge_case_empty_input(self):
        with self.assertRaises(ValueError):
            self.gda.fit(np.array([]), np.array([]))

    def test_edge_case_single_class(self):
        X_single_class = self.X_train[self.y_train == 0]
        y_single_class = self.y_train[self.y_train == 0]
        self.gda.fit(X_single_class, y_single_class)
        y_pred = self.gda.predict(self.X_test)
        self.assertTrue(all(pred == 0 for pred in y_pred))

if __name__ == '__main__':
    unittest.main()
