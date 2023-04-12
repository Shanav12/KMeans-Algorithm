import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_blobs

# calculates the euclidean distance between 2 data points
def distance(first_point, second_point):
    return math.sqrt(np.sum((first_point - second_point) ** 2))

class KMeans:
    # Initializes the model with K's default value being 10 as well as the maximum iterations being 25
    def __init__(self, K = 10, max_iterations = 25, should_plot = False):
        self.K = K
        self.max_iterations = max_iterations
        self.should_plot = should_plot
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.num_samples, self.num_features = X.shape
        # initialization of the centroids
        random_sample_indexes = np.random.choice(self.num_samples, self.K, replace=False)
        self.centroids = [self.X[index] for index in random_sample_indexes]
        # optimization for each clusters
        for _ in range(self.max_iterations):
            # assign samples to closest centroids (create clusters)
            self.clusters = self.create_clusters(self.centroids)

            if self.should_plot:
                self.plot()
            # calculates the new centroids using the clusters
            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)
            #determines whether the old and new centroid are the same
            if self.converges(centroids_old, self.centroids):
                break
            if self.should_plot:
                self.plot()
        # classifies each sample using the index of their clusters
        return self.get_labels(self.clusters)

    def get_labels(self, clusters):
        # assigns a label to each cluster
        labels = np.empty(self.num_samples)
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index
        return labels

    def create_clusters(self, centroids):
        # assigns a sample to the closest centroid
        clusters = [[] for _ in range(self.K)]
        for index, sample in enumerate(self.X):
            centroid_index= self.nearest_centroid(sample, centroids)
            clusters[centroid_index].append(index)
        return clusters

    def nearest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [distance(sample, point) for point in centroids]
        nearest_index = np.argmin(distances)
        return nearest_index

    def get_centroids(self, clusters):
        # assigns the average value of the clusters to centroids
        centroids = np.zeros((self.K, self.num_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0)
            centroids[cluster_index] = cluster_mean
        return centroids

    def converges(self, old_centroids, centroids):
        # distance between each old centroid and new centroid
        distances = [distance(old_centroids[index], centroids[index]) for index in range(self.K)]
        return sum(distances) == 0

    # plots the data points as well as the centroids, shows how the centroids move over each iteration
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for points in self.centroids:
            # plots each centroid marked by a thick black X
            ax.scatter(*points, marker= "X", color = "black", linewidth = 2)
        plt.show()


# Test run of the model
if __name__ == "__main__":
    np.random.seed(35)
    X, y = make_blobs(centers = 4, n_samples = 500, n_features = 2, shuffle = True, random_state = 23)
    #prints out the number of samples and centroids the model will be using
    print(X.shape)
    clusters = len(np.unique(y))
    print(clusters)
    model = KMeans(K = clusters, max_iterations = 150, should_plot = True)
    y_pred = model.predict(X)
    model.plot()
