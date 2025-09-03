# nb_gaussian_multi_csv.py
# Run: python nb_gaussian_multi_csv.py
# What it does:
# - reads multiple CSVs listed in CSV_FILES
# - stacks them together
# - expects numeric features + 'label' column
# - trains + evaluates + plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

CSV_FILES  = ["jan.csv", "feb.csv", "mar.csv"]  # <- change to your files
TARGET_COL = "label"

def load_many(files, target_col):
    frames = []
    for f in files:
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSVs.")

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).dropna()
    y = df.loc[X.index, target_col]
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("No numeric features / rows after cleaning.")
    y = pd.Categorical(y).codes
    return X.values, y

def to_2d(X):
    return X if X.shape[1] == 2 else PCA(n_components=2, random_state=0).fit_transform(X)

if __name__ == "__main__":
    # 1) load
    X, y = load_many(CSV_FILES, TARGET_COL)

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

    # 5) plot
    X2 = to_2d(X)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=y, s=20)
    plt.title(f"Gaussian Naive Bayes on {len(CSV_FILES)} CSVs")
    plt.xlabel("PC1 / X1"); plt.ylabel("PC2 / X2")
    plt.show()
