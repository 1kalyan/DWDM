# kmedoids_sample.py
# Run: python kmedoids_sample.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids


def make_sample(n=300, seed=21):
    rng = np.random.RandomState(seed)
    A = rng.normal([-2,-1], 0.5, size=(n//3,2))
    B = rng.normal([ 2,-1], 0.5, size=(n//3,2))
    C = rng.normal([ 0, 2], 0.5, size=(n//3,2))
    return np.vstack([A,B,C])

def to_2d(X): return X if X.shape[1]==2 else PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
    X = make_sample()
    K = 3
    model = KMedoids(n_clusters=K, method="pam", random_state=0)
    labels = model.fit_predict(X)
    X2 = to_2d(X)
    # medoids are actual data points; take their coordinates
    medoid_points = X[model.medoid_indices_]
    medoids2 = to_2d(medoid_points)

    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=labels)
    plt.scatter(medoids2[:,0], medoids2[:,1], marker="D", s=100)
    plt.title(f"K-Medoids (k={K})")
    plt.xlabel("PC1 / X"); plt.ylabel("PC2 / Y")
    plt.show()
