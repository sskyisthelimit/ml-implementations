import numpy as np


class KNN:
    def __init__(self, k, initial_values, initial_labels):
        self.k = k
        self.initial_values = np.array(initial_values)
        self.initial_labels = np.array(initial_labels)

    def _calculate_distances(self, x):
        self.indexed_distances = []
        for idx, point in enumerate(self.initial_values):
            dist = np.linalg.norm(x - point)
            self.indexed_distances.append([dist, idx])
        self.indexed_k_distances = sorted(self.indexed_distances, key=lambda x: x[0])[:self.k]

    def _determine_class(self):
        class_counts = {}
        for _, idx in self.indexed_k_distances:
            label = self.initial_labels[idx]
            if label not in class_counts:
                class_counts[label] = 1
            else:
                class_counts[label] += 1
        total_distances = {label: 0 for label in class_counts}
        for dist, idx in self.indexed_k_distances:
            total_distances[self.initial_labels[idx]] += dist
        self.predicted_class = min(total_distances, key=total_distances.get)

    def predict(self, X):
        predictions = []
        for x in X:
            self._calculate_distances(x)
            self._determine_class()
            predictions.append(self.predicted_class)
        return np.array(predictions)
