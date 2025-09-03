# minibatch_sample.py
# Run: python minibatch_sample.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def make_sample(n=600, seed=7):
    rng = np.random.RandomState(seed)
    A = rng.normal([0,0], 0.6, size=(n//3,2))
    B = rng.normal([4,0], 0.6, size=(n//3,2))
    C = rng.normal([2,3], 0.6, size=(n//3,2))
    return np.vstack([A,B,C])

def to_2d(X): return X if X.shape[1]==2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = make_sample()
    K = 3
    model = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=64)
    labels = model.fit_predict(X)
    X2, centers2 = to_2d(X), to_2d(model.cluster_centers_)

    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=labels)
    plt.scatter(centers2[:,0], centers2[:,1], marker="x", s=120)
    plt.title(f"MiniBatch K-Means (k={K})")
    plt.xlabel("PC1 / X"); plt.ylabel("PC2 / Y")
    plt.show()
