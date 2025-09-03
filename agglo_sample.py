# agglo_sample.py
# Run: python agglo_sample.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


def make_sample(n=300, seed=13):
    rng = np.random.RandomState(seed)
    A = rng.normal(loc=[-2, -2], scale=0.4, size=(n//3, 2))
    B = rng.normal(loc=[2, -2], scale=0.4, size=(n//3, 2))
    C = rng.normal(loc=[0, 2], scale=0.4, size=(n//3, 2))
    return np.vstack([A, B, C])

def to_2d(X):
    return X if X.shape[1] == 2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = make_sample()
    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(X)
    X2 = to_2d(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels)
    plt.title("Agglomerative Clustering (n_clusters=3)")
    plt.xlabel("PC1 / X")
    plt.ylabel("PC2 / Y")
    plt.show()
