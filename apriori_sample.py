# apriori_sample.py
# Run: python apriori_sample.py

from collections import defaultdict
from itertools import combinations

# ---------- Helper functions ----------

def apriori(transactions, min_support=0.3):
    """
    transactions: list[set[str]]
    min_support: between 0 and 1
    returns: dict{frozenset(items): support}
    """
    n = len(transactions)
    if n == 0:
        return {}

    def support(count):  # convert count -> support
        return count / n

    # L1: count single items
    item_count = defaultdict(int)
    for t in transactions:
        for x in t:
            item_count[x] += 1

    L = {}  # level -> {itemset: count}
    L[1] = {frozenset([i]): c for i, c in item_count.items() if support(c) >= min_support}

    k = 2
    while True:
        prev = L.get(k - 1, {})
        if not prev:
            break

        prev_sets = list(prev.keys())
        # join step: make candidates of size k
        Ck = set()
        for i in range(len(prev_sets)):
            for j in range(i + 1, len(prev_sets)):
                union = prev_sets[i] | prev_sets[j]
                if len(union) == k:
                    # prune: all (k-1) subsets must be frequent
                    all_freq = all(frozenset(s) in prev for s in combinations(union, k - 1))
                    if all_freq:
                        Ck.add(union)

        # count candidates
        Ck_count = defaultdict(int)
        for t in transactions:
            for cand in Ck:
                if cand.issubset(t):
                    Ck_count[cand] += 1

        # keep frequent
        L[k] = {iset: cnt for iset, cnt in Ck_count.items() if support(cnt) >= min_support}
        if not L[k]:
            del L[k]
            break
        k += 1

    # flatten to {itemset: support}
    out = {}
    for level in L.values():
        for iset, cnt in level.items():
            out[iset] = support(cnt)
    return out

def generate_rules(frequent_itemsets, min_confidence=0.6):
    """
    frequent_itemsets: dict{frozenset(items): support}
    returns: list of rule dicts
    """
    rules = []
    for itemset, s_xy in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for A in combinations(items, r):
                A = frozenset(A)
                B = itemset - A
                if not B:
                    continue
                s_x = frequent_itemsets.get(A, 0.0)
                s_y = frequent_itemsets.get(B, 0.0)
                if s_x == 0.0 or s_y == 0.0:
                    continue
                conf = s_xy / s_x
                if conf >= min_confidence:
                    lift = conf / s_y
                    rules.append({
                        "antecedent": tuple(sorted(A)),
                        "consequent": tuple(sorted(B)),
                        "support": s_xy,
                        "confidence": conf,
                        "lift": lift
                    })
    # sort: confidence desc, then lift, then support
    rules.sort(key=lambda r: (r["confidence"], r["lift"], r["support"]), reverse=True)
    return rules

# ---------- Sample data ----------
transactions = [
    {"milk", "bread", "eggs"},
    {"bread", "butter"},
    {"milk", "diapers", "beer", "bread"},
    {"bread", "eggs"},
    {"milk", "diapers", "bread", "butter"},
    {"diapers", "beer"},
    {"milk", "bread"},
]

if __name__ == "__main__":
    minsup = 0.3
    minconf = 0.6

    freq = apriori(transactions, minsup)
    print("# Frequent Itemsets (Apriori, sample)")
    for iset, s in sorted(freq.items(), key=lambda kv: (len(kv[0]), kv[1], tuple(sorted(kv[0])))):
        print(f"{sorted(list(iset))}  support={s:.3f}")
    print()

    rules = generate_rules(freq, minconf)
    print("# Association Rules (Apriori, sample)")
    if not rules:
        print("No rules found.")
    else:
        for r in rules:
            A = ", ".join(r["antecedent"])
            B = ", ".join(r["consequent"])
            print(f"{A} -> {B} | support={r['support']:.3f}, confidence={r['confidence']:.3f}, lift={r['lift']:.3f}")
