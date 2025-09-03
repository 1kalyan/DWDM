# id3_sample.py
# Run: python id3_sample.py
# - Pure Python/Numpy/Pandas implementation of ID3 (information gain)
# - Handles numeric (uses best threshold) and categorical features
# - Prints a readable tree, accuracy, and confusion matrix

from math import log2

import numpy as np
import pandas as pd

# ---------------- ID3 core ----------------

class Node:
    def __init__(self, *, feature=None, threshold=None, children=None, is_leaf=False, prediction=None):
        self.feature = feature          # column name (str)
        self.threshold = threshold      # float for numeric split (<= threshold vs > threshold)
        self.children = children or {}  # dict[value] -> Node for categorical; for numeric: {"<=": Node, ">": Node}
        self.is_leaf = is_leaf
        self.prediction = prediction    # label at leaf

def entropy(y):
    if len(y) == 0:
        return 0.0
    values, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

def information_gain(parent_y, splits):
    """splits: list of 1D arrays of y (the child subsets)"""
    total = len(parent_y)
    if total == 0:
        return 0.0
    H_parent = entropy(parent_y)
    H_children = 0.0
    for child in splits:
        w = len(child) / total
        H_children += w * entropy(child)
    return H_parent - H_children

def best_numeric_threshold(x, y):
    """Find the best threshold (midpoint between sorted unique values) maximizing information gain."""
    # unique sorted values
    vals = np.unique(x[~pd.isna(x)])
    if len(vals) <= 1:
        return None, 0.0
    # candidate thresholds: midpoints
    candidates = (vals[:-1] + vals[1:]) / 2.0
    best_gain = -1.0
    best_t = None
    for t in candidates:
        left_mask = x <= t
        right_mask = x > t
        y_left = y[left_mask]
        y_right = y[right_mask]
        gain = information_gain(y, [y_left, y_right])
        if gain > best_gain:
            best_gain, best_t = gain, t
    return best_t, best_gain

def majority_label(y):
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]

def id3_train(df, target_col="label", max_depth=None, min_samples_split=2):
    features = [c for c in df.columns if c != target_col]
    X = df[features]
    y = df[target_col].to_numpy()

    def build(sub_X, sub_y, depth):
        # stopping conditions
        if len(np.unique(sub_y)) == 1:
            return Node(is_leaf=True, prediction=sub_y[0])
        if (max_depth is not None and depth >= max_depth) or len(sub_y) < min_samples_split or sub_X.shape[1] == 0:
            return Node(is_leaf=True, prediction=majority_label(sub_y))

        best_feature = None
        best_gain = -1.0
        best_split = None
        best_is_numeric = False
        best_threshold = None

        for col in sub_X.columns:
            col_values = sub_X[col]
            if pd.api.types.is_numeric_dtype(col_values):
                # numeric: find threshold
                t, gain = best_numeric_threshold(col_values.to_numpy(), sub_y)
                if t is not None and gain > best_gain:
                    best_gain = gain
                    best_feature = col
                    best_is_numeric = True
                    best_threshold = t
            else:
                # categorical: split by unique values
                splits = []
                for v in col_values.dropna().unique():
                    splits.append(sub_y[col_values == v])
                # Account for NaN as its own branch if exists
                if col_values.isna().any():
                    splits.append(sub_y[col_values.isna()])
                gain = information_gain(sub_y, splits)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = col
                    best_is_numeric = False
                    best_threshold = None

        if best_feature is None or best_gain <= 0.0:
            return Node(is_leaf=True, prediction=majority_label(sub_y))

        # create children
        if best_is_numeric:
            node = Node(feature=best_feature, threshold=best_threshold)
            left_mask = sub_X[best_feature] <= best_threshold
            right_mask = sub_X[best_feature] > best_threshold
            node.children["<="] = build(sub_X[left_mask].drop(columns=[best_feature]),
                                        sub_y[left_mask], depth + 1)
            node.children[">"] = build(sub_X[right_mask].drop(columns=[best_feature]),
                                       sub_y[right_mask], depth + 1)
            return node
        else:
            node = Node(feature=best_feature)
            col_vals = sub_X[best_feature]
            # branches for each category value
            for v in col_vals.dropna().unique():
                mask = col_vals == v
                node.children[v] = build(sub_X[mask].drop(columns=[best_feature]),
                                         sub_y[mask], depth + 1)
            # optional branch for NaN
            if col_vals.isna().any():
                mask = col_vals.isna()
                node.children["<NA>"] = build(sub_X[mask].drop(columns=[best_feature]),
                                              sub_y[mask], depth + 1)
            return node

    return build(X, y, depth=0)

def id3_predict_one(node, row):
    while not node.is_leaf:
        if node.threshold is not None:
            # numeric
            val = row.get(node.feature)
            if pd.isna(val):
                # missing -> pick the larger subtree majority by going to the child with more leaves if possible
                # (simple fallback: go to "<=")
                node = node.children.get("<=")
            else:
                node = node.children["<="] if val <= node.threshold else node.children[">"]
        else:
            # categorical
            val = row.get(node.feature)
            if pd.isna(val):
                nxt = node.children.get("<NA>")
                if nxt is None:
                    # fallback: first child
                    nxt = next(iter(node.children.values()))
                node = nxt
            else:
                nxt = node.children.get(val)
                if nxt is None:
                    # unseen category -> fallback to majority branch
                    nxt = next(iter(node.children.values()))
                node = nxt
    return node.prediction

def id3_predict(node, df_features):
    preds = []
    for _, r in df_features.iterrows():
        preds.append(id3_predict_one(node, r))
    return np.array(preds)

def confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[idx[yt], idx[yp]] += 1
    return labels, cm

# -------------- Sample data --------------

def make_sample():
    # Simple mix of numeric + categorical
    data = {
        "age":      [25, 32, 47, 51, 62, 23, 43, 52, 36, 44, 27, 48],
        "income":   [30, 60, 85, 70, 90, 25, 65, 88, 55, 77, 29, 66],
        "student":  ["no","no","no","no","no","yes","yes","no","yes","yes","yes","no"],
        "credit":   ["fair","excellent","fair","fair","excellent","fair","fair","excellent","excellent","fair","excellent","fair"],
        "label":    ["no","no","yes","yes","yes","no","yes","yes","no","yes","no","yes"],
    }
    return pd.DataFrame(data)

def train_test_split_df(df, test_size=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(len(df) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

def print_tree(node, indent=""):
    if node.is_leaf:
        print(indent + f"Leaf -> predict: {node.prediction}")
        return
    if node.threshold is not None:
        print(indent + f"[NUM] {node.feature} <= {node.threshold:.4f}?")
        print(indent + "  <=:")
        print_tree(node.children['<='], indent + "    ")
        print(indent + "  > :")
        print_tree(node.children['>'], indent + "    ")
    else:
        print(indent + f"[CAT] {node.feature}")
        for val, child in node.children.items():
            print(indent + f"  = {val}:")
            print_tree(child, indent + "    ")

if __name__ == "__main__":
    df = make_sample()
    target = "label"
    train_df, test_df = train_test_split_df(df, test_size=0.3, seed=42)

    tree = id3_train(train_df, target_col=target, max_depth=None, min_samples_split=2)

    print("# Learned ID3 tree:")
    print_tree(tree)
    print()

    y_true = test_df[target].to_numpy()
    X_test = test_df.drop(columns=[target])
    y_pred = id3_predict(tree, X_test)

    acc = (y_true == y_pred).mean()
    print(f"Accuracy: {acc:.3f}")
    labels, cm = confusion_matrix(y_true, y_pred)
    print("Labels:", list(labels))
    print("Confusion matrix:\n", cm)
