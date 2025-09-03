# agglo_one_csv.py
# Run: python agglo_one_csv.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

CSV_FILENAME = "data.csv"

def load_numeric(path):
    df = pd.read_csv(path)
    X = df.select_dtypes(include=[np.number]).dropna()
    if X.shape[1] == 0:
        raise ValueError("No numeric columns in CSV.")
    return X.values

def to_2d(X):
    return X if X.shape[1] == 2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = load_numeric(CSV_FILENAME)
    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(X)
    X2 = to_2d(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels)
    plt.title(f"Agglomerative on {CSV_FILENAME}")
    plt.xlabel("PC1 / X")
    plt.ylabel("PC2 / Y")
    plt.show()
