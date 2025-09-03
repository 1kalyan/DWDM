# dbscan_multi_csv.py
# Run: python dbscan_multi_csv.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

CSV_FILES = ["jan.csv", "feb.csv", "mar.csv"]

def load_many(files):
    frames = [pd.read_csv(f) for f in files]
    df_all = pd.concat(frames, ignore_index=True)
    X = df_all.select_dtypes(include=[np.number]).dropna()
    if X.shape[1] == 0:
        raise ValueError("No numeric columns across CSVs.")
    return X.values

def to_2d(X):
    return X if X.shape[1] == 2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = load_many(CSV_FILES)
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X)
    X2 = to_2d(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels)
    plt.title(f"DBSCAN on {len(CSV_FILES)} CSVs")
    plt.xlabel("PC1 / X")
    plt.ylabel("PC2 / Y")
    plt.show()
