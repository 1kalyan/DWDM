# isolationforest_sample.py
# Run: python isolationforest_sample.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

np.random.seed(42)
X = np.random.normal(50, 5, (200, 2))
X = np.vstack([X, [[100, 100], [105, 110], [110, 120]]])

clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X)
labels = clf.predict(X)

outliers = np.where(labels == -1)[0]
print("ISOLATION FOREST")
print("Outlier indices:", outliers)
print("Outlier values:\n", X[outliers])

plt.figure()
plt.scatter(X[:, 0], X[:, 1], label="Normal")
plt.scatter(X[outliers, 0], X[outliers, 1], color="red", label="Outliers")
plt.title("Outlier Detection (Isolation Forest)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
