# zscore_multi_csv.py
# Run: python zscore_multi_csv.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_FILES = ["jan.csv", "feb.csv", "mar.csv"]
THRESHOLD = 3

frames = [pd.read_csv(f) for f in CSV_FILES]
df = pd.concat(frames, ignore_index=True)
numeric_df = df.select_dtypes(include=[np.number])

for col in numeric_df.columns:
    data = numeric_df[col].values
    mean = np.mean(data)
    std = np.std(data)
    z = (data - mean) / std
    outliers = np.where(np.abs(z) > THRESHOLD)[0]
    print(f"\nColumn: {col}")
    print("Outlier indices:", outliers)
    print("Outlier values:", data[outliers])

    plt.figure()
    plt.scatter(range(len(data)), data, label="Data")
    plt.scatter(outliers, data[outliers], color="red", label="Outliers")
    plt.axhline(mean, color="green", linestyle="--", label="Mean")
    plt.title(f"Outlier Detection (Z-Score) — {col}")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.legend()
    plt.show()
