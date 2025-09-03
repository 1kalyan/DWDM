# lof_sample.py
# Run: python lof_sample.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)
X = np.random.normal(0, 1, (200, 2))
X = np.vstack([X, [[8, 8], [10, 10], [12, 12]]])

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = lof.fit_predict(X)

outliers = np.where(labels == -1)[0]
print("LOCAL OUTLIER FACTOR")
print("Outlier indices:", outliers)
print("Outlier values:\n", X[outliers])

plt.figure()
plt.scatter(X[:, 0], X[:, 1], label="Normal")
plt.scatter(X[outliers, 0], X[outliers, 1], color="red", label="Outliers")
plt.title("Outlier Detection (LOF)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
