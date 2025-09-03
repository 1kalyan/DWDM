# divisive_multi_csv.py
# Run: python divisive_multi_csv.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

CSV_FILES = ["jan.csv", "feb.csv", "mar.csv"]
N_CLUSTERS = 3

def divisive_clustering(X, n_clusters=3):
    clusters = [np.arange(len(X))]
    labels = np.zeros(len(X), dtype=int)

    while len(clusters) < n_clusters:
        variances = [np.var(X[idx], axis=0).sum() for idx in clusters]
        split_idx = np.argmax(variances)
        idx_to_split = clusters.pop(split_idx)
        points_to_split = X[idx_to_split]

        kmeans = KMeans(n_clusters=2, random_state=42)
        new_labels = kmeans.fit_predict(points_to_split)

        clusters.append(idx_to_split[new_labels == 0])
        clusters.append(idx_to_split[new_labels == 1])

    final_labels = np.zeros(len(X), dtype=int)
    for cluster_id, idx in enumerate(clusters):
        final_labels[idx] = cluster_id
    return final_labels

def to_2d(X):
    return X if X.shape[1] == 2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    frames = [pd.read_csv(f) for f in CSV_FILES]
    df = pd.concat(frames, ignore_index=True)
    X = df.select_dtypes(include=[np.number]).dropna().values

    labels = divisive_clustering(X, N_CLUSTERS)
    X2 = to_2d(X)

    plt.figure(figsize=(7, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="viridis", s=30)
    plt.title(f"Divisive Clustering on {len(CSV_FILES)} CSVs")
    plt.xlabel("PC1 / X")
    plt.ylabel("PC2 / Y")
    plt.show()
