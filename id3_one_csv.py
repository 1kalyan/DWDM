# id3_one_csv.py
# Run: python id3_one_csv.py
# CSV rules:
# - Put e.g. data.csv beside this file
# - Must contain a target column named 'label' (change TARGET below if needed)
# - Can mix numeric and categorical columns

from math import log2

import numpy as np
import pandas as pd

CSV_FILENAME = "data.csv"
TARGET = "label"

# ---------- (same ID3 core as sample, compact copy) ----------

class Node:
    def __init__(self, *, feature=None, threshold=None, children=None, is_leaf=False, prediction=None):
        self.feature=feature; self.threshold=threshold
        self.children=children or {}; self.is_leaf=is_leaf; self.prediction=prediction

def entropy(y):
    if len(y)==0: return 0.0
    vals, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    return -np.sum(p*np.log2(p))

def information_gain(parent_y, splits):
    total=len(parent_y)
    if total==0: return 0.0
    H=entropy(parent_y); hc=0.0
    for s in splits:
        w=len(s)/total; hc += w*entropy(s)
    return H - hc

def best_numeric_threshold(x, y):
    vals = np.unique(x[~pd.isna(x)])
    if len(vals)<=1: return (None, 0.0)
    cand=(vals[:-1]+vals[1:])/2.0
    best_t=None; best_g=-1.0
    for t in cand:
        yl=y[x<=t]; yr=y[x>t]
        g=information_gain(y,[yl,yr])
        if g>best_g: best_g=g; best_t=t
    return best_t,best_g

def majority_label(y):
    v,c=np.unique(y,return_counts=True)
    return v[np.argmax(c)]

def id3_train(df, target_col="label", max_depth=None, min_samples_split=2):
    feats=[c for c in df.columns if c!=target_col]
    X=df[feats]; y=df[target_col].to_numpy()
    def build(SX, Sy, depth):
        if len(np.unique(Sy))==1: return Node(is_leaf=True,prediction=Sy[0])
        if (max_depth is not None and depth>=max_depth) or len(Sy)<min_samples_split or SX.shape[1]==0:
            return Node(is_leaf=True,prediction=majority_label(Sy))
        best_f=None; best_g=-1.0; best_num=False; best_t=None
        for col in SX.columns:
            colv=SX[col]
            if pd.api.types.is_numeric_dtype(colv):
                t,g=best_numeric_threshold(colv.to_numpy(),Sy)
                if t is not None and g>best_g: best_f=col; best_g=g; best_num=True; best_t=t
            else:
                splits=[Sy[colv==v] for v in colv.dropna().unique()]
                if colv.isna().any(): splits.append(Sy[colv.isna()])
                g=information_gain(Sy,splits)
                if g>best_g: best_f=col; best_g=g; best_num=False; best_t=None
        if best_f is None or best_g<=0.0: return Node(is_leaf=True,prediction=majority_label(Sy))
        if best_num:
            node=Node(feature=best_f,threshold=best_t)
            L=SX[best_f]<=best_t; R=SX[best_f]>best_t
            node.children["<="]=build(SX[L].drop(columns=[best_f]),Sy[L],depth+1)
            node.children[">"]=build(SX[R].drop(columns=[best_f]),Sy[R],depth+1)
            return node
        else:
            node=Node(feature=best_f)
            colv=SX[best_f]
            for v in colv.dropna().unique():
                m=colv==v
                node.children[v]=build(SX[m].drop(columns=[best_f]),Sy[m],depth+1)
            if colv.isna().any():
                m=colv.isna()
                node.children["<NA>"]=build(SX[m].drop(columns=[best_f]),Sy[m],depth+1)
            return node
    return build(X,y,0)

def id3_predict_one(node, row):
    while not node.is_leaf:
        if node.threshold is not None:
            val=row.get(node.feature)
            node = node.children["<="] if (not pd.isna(val) and val<=node.threshold) else node.children.get(">", next(iter(node.children.values())))
        else:
            val=row.get(node.feature)
            node=node.children.get("<NA>") if pd.isna(val) else node.children.get(val, next(iter(node.children.values())))
    return node.prediction

def id3_predict(node, X):
    return np.array([id3_predict_one(node, r) for _,r in X.iterrows()])

def confusion_matrix(y_true, y_pred):
    labs=np.unique(np.concatenate([y_true,y_pred])); idx={v:i for i,v in enumerate(labs)}
    cm=np.zeros((len(labs),len(labs)),dtype=int)
    for yt,yp in zip(y_true,y_pred): cm[idx[yt],idx[yp]]+=1
    return labs, cm

def train_test_split_df(df, test_size=0.25, seed=0):
    rng=np.random.default_rng(seed); idx=np.arange(len(df)); rng.shuffle(idx)
    s=int(len(df)*(1-test_size)); tr,te=idx[:s],idx[s:]
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)

def print_tree(node, indent=""):
    if node.is_leaf:
        print(indent+f"Leaf -> predict: {node.prediction}"); return
    if node.threshold is not None:
        print(indent+f"[NUM] {node.feature} <= {node.threshold:.4f}?")
        print(indent+"  <=:"); print_tree(node.children['<='], indent+"    ")
        print(indent+"  > :"); print_tree(node.children['>'],  indent+"    ")
    else:
        print(indent+f"[CAT] {node.feature}")
        for v,ch in node.children.items():
            print(indent+f"  = {v}:"); print_tree(ch, indent+"    ")

# ----------------- Run -----------------

if __name__ == "__main__":
    df = pd.read_csv(CSV_FILENAME)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    train_df, test_df = train_test_split_df(df, test_size=0.25, seed=42)
    tree = id3_train(train_df, target_col=TARGET, max_depth=None, min_samples_split=2)

    print("# Learned ID3 tree:")
    print_tree(tree); print()

    y_true = test_df[TARGET].to_numpy()
    X_test = test_df.drop(columns=[TARGET])
    y_pred = id3_predict(tree, X_test)

    acc = (y_true == y_pred).mean()
    print(f"Accuracy: {acc:.3f}")
    labels, cm = confusion_matrix(y_true, y_pred)
    print("Labels:", list(labels))
    print("Confusion matrix:\n", cm)
