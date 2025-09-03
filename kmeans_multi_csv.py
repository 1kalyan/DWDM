# kmeans_multi_csv.py
# Update CSV_FILES with files in the same folder; they will be concatenated vertically.
# Run: python kmeans_multi_csv.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

CSV_FILES = ["jan.csv", "feb.csv", "mar.csv"]  # change as needed
K = 3

def load_many(files):
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    X = df_all.select_dtypes(include=[np.number]).dropna()
    if X.shape[1] == 0:
        raise ValueError("No numeric columns found across CSVs.")
    return X.values

def to_2d(X):
    return X if X.shape[1] == 2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = load_many(CSV_FILES)
    model = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = model.fit_predict(X)
    X2 = to_2d(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels)
    centers2 = to_2d(model.cluster_centers_)
    plt.scatter(centers2[:, 0], centers2[:, 1], marker="x", s=120)
    plt.title(f"K-Means (k={K}) on {len(CSV_FILES)} CSVs")
    plt.xlabel("PC1 / X")
    plt.ylabel("PC2 / Y")
    plt.show()
