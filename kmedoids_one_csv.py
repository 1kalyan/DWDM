# kmedoids_one_csv.py
# Run: python kmedoids_one_csv.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

CSV_FILENAME = "data.csv"
K = 3

def load_numeric(p):
    df = pd.read_csv(p)
    X = df.select_dtypes(include=[np.number]).dropna()
    if X.shape[1]==0: raise ValueError("No numeric columns found.")
    return X.values

def to_2d(X): return X if X.shape[1]==2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = load_numeric(CSV_FILENAME)
    model = KMedoids(n_clusters=K, method="pam", random_state=0)
    labels = model.fit_predict(X)
    X2 = to_2d(X)
    medoids2 = to_2d(X[model.medoid_indices_])

    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=labels)
    plt.scatter(medoids2[:,0], medoids2[:,1], marker="D", s=100)
    plt.title(f"K-Medoids on {CSV_FILENAME} (k={K})")
    plt.xlabel("PC1 / X"); plt.ylabel("PC2 / Y")
    plt.show()
