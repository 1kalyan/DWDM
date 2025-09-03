# dbscan_sample.py
# Run: python dbscan_sample.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def make_sample(n=400, seed=0):
    rng = np.random.RandomState(seed)
    ring = rng.normal(0, 1, (n//2, 2))
    ring = ring / np.linalg.norm(ring, axis=1, keepdims=True) * (1.0 + 0.1*rng.randn(n//2,1))
    blob = rng.normal(loc=[3, 0], scale=0.3, size=(n//2, 2))
    return np.vstack([ring, blob])

def to_2d(X):
    return X if X.shape[1] == 2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = make_sample()
    model = DBSCAN(eps=0.3, min_samples=5)
    labels = model.fit_predict(X)
    X2 = to_2d(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels)
    plt.title("DBSCAN (eps=0.3, min_samples=5)")
    plt.xlabel("PC1 / X")
    plt.ylabel("PC2 / Y")
    plt.show()
