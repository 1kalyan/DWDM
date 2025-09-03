# fpgrowth_multi_csv.py
# Run: update CSV_FILES below, then:
# python fpgrowth_multi_csv.py

import csv
import os
from collections import defaultdict
from itertools import combinations

CSV_FILES = ["jan.csv", "feb.csv", "mar.csv"]

def load_tx_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    tx = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or all((c or "").strip() == "" for c in row): continue
            items = [c.strip() for c in (row[0].split(",") if len(row)==1 else row)]
            items = [x for x in items if x]
            if items: tx.append(set(items))
    return tx

def load_tx_csvs(paths):
    all_tx = []
    for p in paths:
        all_tx.extend(load_tx_csv(p))
    return all_tx

# --- FP-growth (same as in one_csv, trimmed comments) ---
class Node:
    def __init__(self, item, parent=None):
        self.item=item; self.count=1; self.parent=parent; self.children={}; self.link=None

def header_counts(transactions, min_support):
    n=len(transactions); counts=defaultdict(int)
    for t in transactions:
        for i in t: counts[i]+=1
    min_count=max(1,int(min_support*n+1e-9)) if n else 0
    keep={i:c for i,c in counts.items() if c>=min_count}
    header={i:[c,None] for i,c in keep.items()}
    return keep,n,header

def order_tx(tx,hc):
    it=[i for i in tx if i in hc]; it.sort(key=lambda i:(-hc[i],i)); return it

def insert(root,header,ordered):
    node=root
    for it in ordered:
        if it in node.children:
            node=node.children[it]; node.count+=1
        else:
            new=Node(it,node); node.children[it]=new; node=new
            if header[it][1] is None: header[it][1]=node
            else:
                cur=header[it][1]
                while cur.link is not None: cur=cur.link
                cur.link=node

def build_tree(transactions,min_support):
    hc,n,header=header_counts(transactions,min_support)
    root=Node(None)
    if not hc: return root,{},n
    for t in transactions:
        o=order_tx(t,hc)
        if o: insert(root,header,o)
    return root,header,n

def ascend(node):
    path=[]
    while node and node.parent and node.parent.item is not None:
        node=node.parent; path.append(node.item)
    return list(reversed(path))

def mine(header,min_support,n,suffix,out):
    for item,(count,head) in sorted(header.items(), key=lambda kv:(kv[1][0],kv[0])):
        new_suf=tuple(sorted(suffix+(item,)))
        out[frozenset(new_suf)]=count/n if n else 0.0
        cond=[]; cur=head
        while cur is not None:
            path=ascend(cur)
            if path: cond.extend([set(path)]*cur.count)
            cur=cur.link
        if cond:
            _,ch,_=build_tree(cond,min_support)
            if ch: mine(ch,min_support,len(cond),new_suf,out)

def fpgrowth(transactions,min_support=0.3):
    root,header,n=build_tree(transactions,min_support)
    if not header: return {}
    freq={}; mine(header,min_support,n,tuple(),freq)
    total=defaultdict(int)
    for t in transactions:
        for i in t: total[i]+=1
    for i,c in total.items():
        s=c/n if n else 0.0
        if s>=min_support: freq.setdefault(frozenset([i]),s)
    return freq

def rules(fis,min_conf=0.6):
    out=[]
    for S,sxy in fis.items():
        if len(S)<2: continue
        items=list(S)
        for r in range(1,len(items)):
            from itertools import combinations
            for A in combinations(items,r):
                A=frozenset(A); B=S-A
                sx=fis.get(A,0.0); sy=fis.get(B,0.0)
                if sx==0 or sy==0: continue
                conf=sxy/sx
                if conf>=min_conf:
                    out.append({
                        "antecedent":tuple(sorted(A)),
                        "consequent":tuple(sorted(B)),
                        "support":sxy,
                        "confidence":conf,
                        "lift":conf/sy
                    })
    out.sort(key=lambda r:(r["confidence"],r["lift"],r["support"]),reverse=True)
    return out

if __name__ == "__main__":
    tx = load_tx_csvs(CSV_FILES)
    print(f"Loaded {len(tx)} transactions from {len(CSV_FILES)} CSV files.\n")
    freq = fpgrowth(tx, 0.3)
    print("# Frequent Itemsets (FP-Growth, multi-CSV)")
    for iset, s in sorted(freq.items(), key=lambda kv: (len(kv[0]), kv[1], tuple(sorted(kv[0])))):
        print(f"{sorted(list(iset))}  support={s:.3f}")
    print()
    rs = rules(freq, 0.6)
    print("# Association Rules (FP-Growth, multi-CSV)")
    if not rs: print("No rules found.")
    else:
        for r in rs:
            A = ", ".join(r["antecedent"]); B = ", ".join(r["consequent"])
            print(f"{A} -> {B} | support={r['support']:.3f}, confidence={r['confidence']:.3f}, lift={r['lift']:.3f}")
