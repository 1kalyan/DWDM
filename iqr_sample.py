# iqr_sample.py
# Run: python iqr_sample.py
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.normal(30, 10, 200)
data = np.append(data, [100, 105, 110])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = np.where((data < lower) | (data > upper))[0]

print("IQR METHOD")
print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print("Outlier indices:", outliers)
print("Outlier values:", data[outliers])

plt.figure()
plt.boxplot(data, vert=False)
plt.title("Outlier Detection (IQR Method)")
plt.xlabel("Value")
plt.show()
