# kmeanspp_one_csv.py
# Put data.csv beside this file. Run: python kmeanspp_one_csv.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

CSV_FILENAME = "data.csv"
K = 3

def load_numeric(path):
    df = pd.read_csv(path)
    X = df.select_dtypes(include=[np.number]).dropna()
    if X.shape[1] == 0: raise ValueError("No numeric columns found.")
    return X.values

def to_2d(X): return X if X.shape[1]==2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = load_numeric(CSV_FILENAME)
    model = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=0)
    labels = model.fit_predict(X)
    X2, centers2 = to_2d(X), to_2d(model.cluster_centers_)

    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=labels)
    plt.scatter(centers2[:,0], centers2[:,1], marker="x", s=120)
    plt.title(f"K-Means++ on {CSV_FILENAME} (k={K})")
    plt.xlabel("PC1 / X"); plt.ylabel("PC2 / Y")
    plt.show()
