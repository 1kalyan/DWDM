# nb_gaussian_one_csv.py
# Run: python nb_gaussian_one_csv.py
# CSV requirements (simple):
# - Put a file like data.csv beside this script
# - Must contain numeric feature columns + a target column named 'label'
# Example:
#   f1,f2,f3,label
#   1.2,3.4,0.1,A
#   2.3,1.1,-0.2,B
#
# If you use a different target name, change TARGET_COL below.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

CSV_FILENAME = "data.csv"   # <- change if needed
TARGET_COL   = "label"      # <- change if your label column is named differently

def load_numeric_xy(path, target_col):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")
    # numeric features only
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).dropna()
    y = df.loc[X.index, target_col]
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("No numeric features / rows after cleaning.")
    # Convert labels to simple integers if they are strings
    y = pd.Categorical(y).codes
    return X.values, y

def to_2d(X):
    return X if X.shape[1] == 2 else PCA(n_components=2, random_state=0).fit_transform(X)

if __name__ == "__main__":
    # 1) load
    X, y = load_numeric_xy(CSV_FILENAME, TARGET_COL)

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

    # 5) plot (2D via PCA if needed)
    X2 = to_2d(X)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=y, s=20)
    plt.title(f"Gaussian Naive Bayes on {CSV_FILENAME}")
    plt.xlabel("PC1 / X1"); plt.ylabel("PC2 / X2")
    plt.show()
