# fpgrowth_sample.py
# Run: python fpgrowth_sample.py

from collections import defaultdict
from itertools import combinations


class Node:
    def __init__(self, item, parent=None):
        self.item = item
        self.count = 1
        self.parent = parent
        self.children = {}
        self.link = None  # next node with same item

def build_header_and_counts(transactions, min_support):
    n = len(transactions)
    counts = defaultdict(int)
    for t in transactions:
        for i in t: counts[i] += 1
    min_count = max(1, int(min_support * n + 1e-9)) if n else 0
    frequent = {i: c for i, c in counts.items() if c >= min_count}
    header = {i: [c, None] for i, c in frequent.items()}  # item -> [count, head_node]
    return frequent, n, header

def order_tx(tx, header_counts):
    items = [i for i in tx if i in header_counts]
    # sort by frequency desc, then alphabetically
    items.sort(key=lambda i: (-header_counts[i], i))
    return items

def insert_transaction(root, header, ordered_items):
    node = root
    for it in ordered_items:
        if it in node.children:
            node = node.children[it]
            node.count += 1
        else:
            new_node = Node(it, parent=node)
            node.children[it] = new_node
            node = new_node
            # link into header chain
            if header[it][1] is None:
                header[it][1] = node
            else:
                cur = header[it][1]
                while cur.link is not None:
                    cur = cur.link
                cur.link = node

def build_tree(transactions, min_support):
    header_counts, n, header = build_header_and_counts(transactions, min_support)
    root = Node(None)
    if not header_counts:
        return root, {}, n
    for t in transactions:
        ordered = order_tx(t, header_counts)
        if ordered:
            insert_transaction(root, header, ordered)
    return root, header, n

def ascend_path(node):
    path = []
    while node and node.parent and node.parent.item is not None:
        node = node.parent
        path.append(node.item)
    return list(reversed(path))

def mine_tree(header, min_support, n, suffix, out):
    # items in ascending frequency to build conditional trees
    items_sorted = sorted(header.items(), key=lambda kv: (kv[1][0], kv[0]))
    for item, (count, head) in items_sorted:
        new_suffix = tuple(sorted(suffix + (item,)))
        out[frozenset(new_suffix)] = count / n if n else 0.0

        # build conditional pattern base
        cond_transactions = []
        cur = head
        while cur is not None:
            path = ascend_path(cur)
            if path:
                cond_transactions.extend([set(path)] * cur.count)
            cur = cur.link

        if cond_transactions:
            cond_root, cond_header, m = build_tree(cond_transactions, min_support)
            if cond_header:
                mine_tree(cond_header, min_support, len(cond_transactions), new_suffix, out)

def fpgrowth(transactions, min_support=0.3):
    root, header, n = build_tree(transactions, min_support)
    if not header:
        return {}
    freq = {}
    mine_tree(header, min_support, n, suffix=tuple(), out=freq)

    # ensure singletons exist (in case)
    total = defaultdict(int)
    for t in transactions:
        for i in t: total[i] += 1
    for i, c in total.items():
        s = c / n if n else 0.0
        if s >= min_support:
            freq.setdefault(frozenset([i]), s)
    return freq

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

# ----- Sample data -----
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
    freq = fpgrowth(transactions, minsup)
    print("# Frequent Itemsets (FP-Growth, sample)")
    for iset, s in sorted(freq.items(), key=lambda kv: (len(kv[0]), kv[1], tuple(sorted(kv[0])))):
        print(f"{sorted(list(iset))}  support={s:.3f}")
    print()
    rules = generate_rules(freq, minconf)
    print("# Association Rules (FP-Growth, sample)")
    if not rules: print("No rules found.")
    else:
        for r in rules:
            A = ", ".join(r["antecedent"]); B = ", ".join(r["consequent"])
            print(f"{A} -> {B} | support={r['support']:.3f}, confidence={r['confidence']:.3f}, lift={r['lift']:.3f}")
