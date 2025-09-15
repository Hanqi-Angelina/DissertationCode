from collections import defaultdict
import os, sys 
import pandas as pd
from collections import defaultdict
from itertools import combinations
import numpy as np

def count_three_types(adj_np, var_list, indices_to_keep):
    # count the three types of directed and undirected edges
    sub = adj_np[:indices_to_keep, :indices_to_keep]
    counts = defaultdict(int)
    n = min(len(var_list), sub.shape[0])
    for i in range(n):
        ui = var_list[i]
        for j in range(i+1, n):
            vj = var_list[j]
            a = (int(sub[i, j]), int(sub[j, i]))
            if a == (-1, 1):
                counts[(ui, vj, "u->v")] += 1
            elif a == (1, -1):
                counts[(ui, vj, "v->u")] += 1
            elif a == (-1, -1):
                counts[(ui, vj, "undirected")] += 1
            elif a == (0, 0):
                counts[(ui, vj, "none")] += 1
            else: # other values produced by the algorithm
                counts[(ui, vj, "unknown")] += 1
    return counts

EDGE_TYPES = ("u->v", "v->u", "undirected", "none", "unknown")

def merge_counts(dicts):
    # take in a list of dicts and get their sums
    out = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            out[k] += v
    return dict(out)

def counts_to_df(merged_counts, var_list):
    # full grid of pairs × edge types (fill missing with 0)
    rows = []
    for u, v in combinations(var_list, 2):
        for et in EDGE_TYPES:
            rows.append((u, v, et, merged_counts.get((u, v, et), 0)))
    return pd.DataFrame(rows, columns=["u", "v", "edge_type", "count"])

