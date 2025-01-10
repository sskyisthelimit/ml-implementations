import numpy as np
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, k):
        self.k = k
        self.done_fit = False

    def fit(self, initial_values, initial_labels):
        self.initial_values = np.array(initial_values)
        self.initial_labels = np.array(initial_labels)
        self.done_fit = True

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
        if self.done_fit:
            predictions = []
            for x in X:
                self._calculate_distances(np.array(x))
                self._determine_class()
                predictions.append(self.predicted_class)
            return np.array(predictions)
        else:
            raise ValueError("fit() wasn't called before predict()")


data_points = [
    [10, 2, 6], [8, 3, 5], [7, 2, 8], [10, 3, 8],
    [1, 3, 10], [2, 3, 10], [1, 2, 8], [3, 1, 9],
    [4, 4, 5], [5, 4, 6], [6, 3, 5], [3, 3, 4]
]

data_points_labels = [
    "blue", "blue", "blue", "blue",
    "green", "green", "green", "green",
    "red", "red", "red", "red" 
]

knn = KNN(k=3)
knn.fit(initial_values=data_points,
        initial_labels=data_points_labels)
new_point = [3, 4, 10]
new_point_label = knn.predict(new_point)[0]

class_colors = {
    "blue": "#4c34eb",
    "red": "#eb2121",
    "green": "#007020",
}

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection="3d")

for idx, data_point in enumerate(data_points):
    ax.scatter(
        data_point[0], data_point[1],
        data_point[2], c=class_colors[data_points_labels[idx]],
        s=60)

ax.scatter(
    new_point[0], new_point[1], new_point[2],
    c=class_colors[new_point_label], s=120, zorder=100,
    marker="*"
)

for idx, data_point in enumerate(data_points):
    ax.plot(
        [data_point[0], new_point[0]],
        [data_point[1], new_point[1]],
        [data_point[2], new_point[2]],
        linestyle="--", linewidth=1,
        c=class_colors[data_points_labels[idx]]
    )

plt.show()