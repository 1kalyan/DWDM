# kmeans_one_csv.py
# Put your CSV (e.g., data.csv) in the same folder; numeric columns will be used.
# Run: python kmeans_one_csv.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

CSV_FILENAME = "data.csv"   # change if needed
K = 3                       # number of clusters

def load_numeric(path):
    df = pd.read_csv(path)
    X = df.select_dtypes(include=[np.number]).dropna()
    if X.shape[1] == 0:
        raise ValueError("No numeric columns found in CSV.")
    return X.values

def to_2d(X):
    return X if X.shape[1] == 2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = load_numeric(CSV_FILENAME)
    model = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = model.fit_predict(X)
    X2 = to_2d(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels)
    centers2 = to_2d(model.cluster_centers_)
    plt.scatter(centers2[:, 0], centers2[:, 1], marker="x", s=120)
    plt.title(f"K-Means (k={K}) on {CSV_FILENAME}")
    plt.xlabel("PC1 / X")
    plt.ylabel("PC2 / Y")
    plt.show()
