# nb_gaussian_sample.py
# Run: python nb_gaussian_sample.py
# What it does:
# - makes a tiny 2D numeric dataset with 3 classes
# - trains Gaussian Naive Bayes
# - prints accuracy & plots points (easy to understand)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def make_sample(n=300, seed=42):
    rng = np.random.RandomState(seed)
    A = rng.normal(loc=[0, 0],  scale=0.6, size=(n//3, 2))
    B = rng.normal(loc=[3, 3],  scale=0.6, size=(n//3, 2))
    C = rng.normal(loc=[-3, 3], scale=0.6, size=(n//3, 2))
    X = np.vstack([A, B, C])
    y = np.array([0]*(n//3) + [1]*(n//3) + [2]*(n//3))
    return X, y

if __name__ == "__main__":
    # 1) data
    X, y = make_sample()

    # 2) split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    # 3) model
    clf = GaussianNB()
    clf.fit(Xtr, ytr)

    # 4) eval
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:\n", confusion_matrix(yte, ypred))

    # 5) plot (already 2D)
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y, s=25)
    plt.title("Gaussian Naive Bayes (sample data)")
    plt.xlabel("X1"); plt.ylabel("X2")
    plt.show()
