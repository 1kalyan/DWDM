# zscore_sample.py
# Run: python zscore_sample.py
import matplotlib.pyplot as plt
import numpy as np

# Create sample data with outliers
np.random.seed(42)
data = np.random.normal(50, 5, 200)
data = np.append(data, [100, 105, 110])  # add outliers

threshold = 3  # Z-score threshold

mean = np.mean(data)
std = np.std(data)
z_scores = (data - mean) / std
outliers = np.where(np.abs(z_scores) > threshold)[0]

print("Z-SCORE METHOD")
print(f"Mean: {mean:.2f}, Std: {std:.2f}")
print("Outlier indices:", outliers)
print("Outlier values:", data[outliers])

# Plot
plt.figure()
plt.scatter(range(len(data)), data, label="Data")
plt.scatter(outliers, data[outliers], color="red", label="Outliers")
plt.axhline(mean, color="green", linestyle="--", label="Mean")
plt.title("Outlier Detection (Z-Score)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.show()
