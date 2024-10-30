import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class kMeans:
    def __init__(self, k=3, max_iterations=500, epsilon=1e-5):
        self.k = k
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    @staticmethod
    def euclidean_distance(X, centroids):
        return np.sqrt(np.sum((centroids - X)**2, axis=1))

    @staticmethod
    def init_centroids(self, input):
        centroids = np.random.uniform(
            np.amin(input, axis=0),
            np.amax(input, axis=0),
            size=(1, input.shape[1]))

        for idx in range(1, self.k):
            max_distance = 0
            next_centroid = None

            for data_point in input:
                centroids_distances = self.euclidean_distance(
                    data_point, np.array(centroids))
                if np.max(centroids_distances) > max_distance:
                    max_distance = np.max(centroids_distances)
                    next_centroid = data_point
            
            centroids = np.vstack((centroids, np.array(next_centroid)))
        
        return np.array(centroids)

    def fit(self, input, mode='default'):
        if mode == '++':
            self.centroids = self.init_centroids(self, input)
        elif mode == 'default':
            self.centroids = np.random.uniform(
                np.amin(input, axis=0),
                np.amax(input, axis=0),
                size=(self.k, input.shape[1]))
            
        for _ in range(self.max_iterations):
            centroids_elements_indices = [[] for i in range(self.k)]
            
            for idx, data_point in enumerate(input):
                centroids_distances = self.euclidean_distance(data_point,
                                                              self.centroids)
                centroids_elements_indices[
                    np.argmin(centroids_distances, axis=0)].append(idx) 
            
            new_centroids = []

            for cen_idx in range(self.k):
                if len(centroids_elements_indices[cen_idx]) == 0:
                    new_centroids.append(self.centroids[cen_idx])
                else:
                    indices = centroids_elements_indices[cen_idx]
                    new_centroids.append(np.mean(input[indices], axis=0))

            new_centroids = np.array(new_centroids)
            if np.max(self.centroids - new_centroids) < self.epsilon:
                return centroids_elements_indices
            else:
                self.centroids = new_centroids

        return centroids_elements_indices
    

random_input = np.random.randint(0, 100, (50, 2))
k_means = kMeans()
clustered_indices = k_means.fit(random_input, mode="++")
colors = list(mcolors.TABLEAU_COLORS.keys())[:k_means.k]

for cluster_idx, indices in enumerate(clustered_indices):
    cluster_points = random_input[indices]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=colors[cluster_idx],
                label=f'Cluster {cluster_idx+1}')

plt.scatter(k_means.centroids[:, 0], k_means.centroids[:, 1],
            s=100, color='black', marker="*", label='Centroids')

plt.legend()
plt.show()
