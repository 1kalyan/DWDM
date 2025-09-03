# apriori_multi_csv.py
# Run: put your CSVs in the same folder, update CSV_FILES, then:
# python apriori_multi_csv.py
import csv
import os
from collections import defaultdict
from itertools import combinations

CSV_FILES = ["jan.csv", "feb.csv", "mar.csv"]  # change these names to yours

def load_transactions_from_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    tx = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all((c or "").strip() == "" for c in row):
                continue
            if len(row) == 1:
                items = [c.strip() for c in row[0].split(",")]
            else:
                items = [c.strip() for c in row]
            items = [x for x in items if x]
            if items:
                tx.append(set(items))
    return tx

def load_transactions_from_csvs(paths):
    all_tx = []
    for p in paths:
        all_tx.extend(load_transactions_from_csv(p))
    return all_tx

def apriori(transactions, min_support=0.3):
    n = len(transactions)
    if n == 0: return {}
    def support(cnt): return cnt / n

    item_count = defaultdict(int)
    for t in transactions:
        for x in t: item_count[x] += 1
    L = {}
    L[1] = {frozenset([i]): c for i, c in item_count.items() if support(c) >= min_support}

    k = 2
    while True:
        prev = L.get(k-1, {})
        if not prev: break
        prev_sets = list(prev.keys())
        Ck = set()
        for i in range(len(prev_sets)):
            for j in range(i+1, len(prev_sets)):
                u = prev_sets[i] | prev_sets[j]
                if len(u) == k and all(frozenset(s) in prev for s in combinations(u, k-1)):
                    Ck.add(u)
        Ck_count = defaultdict(int)
        for t in transactions:
            for c in Ck:
                if c.issubset(t): Ck_count[c] += 1
        L[k] = {iset: cnt for iset, cnt in Ck_count.items() if support(cnt) >= min_support}
        if not L[k]:
            del L[k]; break
        k += 1

    out = {}
    for level in L.values():
        for iset, cnt in level.items():
            out[iset] = support(cnt)
    return out

def generate_rules(frequent_itemsets, min_confidence=0.6):
    rules = []
    for itemset, sxy in frequent_itemsets.items():
        if len(itemset) < 2: continue
        items = list(itemset)
        for r in range(1, len(items)):
            for A in combinations(items, r):
                A = frozenset(A)
                B = itemset - A
                if not B: continue
                sx = frequent_itemsets.get(A, 0.0)
                sy = frequent_itemsets.get(B, 0.0)
                if sx == 0.0 or sy == 0.0: continue
                conf = sxy / sx
                if conf >= min_confidence:
                    rules.append({
                        "antecedent": tuple(sorted(A)),
                        "consequent": tuple(sorted(B)),
                        "support": sxy,
                        "confidence": conf,
                        "lift": conf / sy
                    })
    rules.sort(key=lambda r: (r["confidence"], r["lift"], r["support"]), reverse=True)
    return rules

if __name__ == "__main__":
    tx = load_transactions_from_csvs(CSV_FILES)
    print(f"Loaded {len(tx)} transactions from {len(CSV_FILES)} CSV files.\n")
    freq = apriori(tx, min_support=0.3)
    print("# Frequent Itemsets (Apriori, multi-CSV)")
    for iset, s in sorted(freq.items(), key=lambda kv: (len(kv[0]), kv[1], tuple(sorted(kv[0])))):
        print(f"{sorted(list(iset))}  support={s:.3f}")
    print()
    rules = generate_rules(freq, min_confidence=0.6)
    print("# Association Rules (Apriori, multi-CSV)")
    if not rules: print("No rules found.")
    else:
        for r in rules:
            A = ", ".join(r["antecedent"]); B = ", ".join(r["consequent"])
            print(f"{A} -> {B} | support={r['support']:.3f}, confidence={r['confidence']:.3f}, lift={r['lift']:.3f}")
