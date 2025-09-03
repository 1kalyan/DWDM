# divisive_sample.py
# Run: python divisive_sample.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def divisive_clustering(X, n_clusters=3):
    # Start with all points in one cluster
    clusters = [np.arange(len(X))]
    labels = np.zeros(len(X), dtype=int)

    while len(clusters) < n_clusters:
        # Choose the cluster with the largest variance to split
        variances = [np.var(X[idx], axis=0).sum() for idx in clusters]
        split_idx = np.argmax(variances)

        # Get the points in the chosen cluster
        idx_to_split = clusters.pop(split_idx)
        points_to_split = X[idx_to_split]

        # Split using KMeans with k=2
        kmeans = KMeans(n_clusters=2, random_state=42)
        new_labels = kmeans.fit_predict(points_to_split)

        # Assign new labels within global array
        labels[idx_to_split[new_labels == 1]] = len(clusters) + 1
        clusters.append(idx_to_split[new_labels == 1])
        clusters.append(idx_to_split[new_labels == 0])

    # Final relabeling to ensure sequential cluster labels
    final_labels = np.zeros(len(X), dtype=int)
    for cluster_id, idx in enumerate(clusters):
        final_labels[idx] = cluster_id

    return final_labels

if __name__ == "__main__":
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

    # Perform divisive clustering
    n_clusters = 3
    labels = divisive_clustering(X, n_clusters)

    # Plot results
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
    plt.title(f"Divisive Clustering (Top-Down) - {n_clusters} Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
