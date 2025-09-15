import pandas as pd
import numpy as np
from StructureLearning.RLCD.RLCD_alg import RLCD
from utils.Chi2RankTest import Chi2RankTest
from utils.GraphDrawer import DotGraph
from collections import defaultdict
import itertools
import json
import os

# === CONFIG ===
df = pd.read_csv("Data/Teacher_Burnout_data.csv")
df = df.rename(columns=lambda c: "X_" + c)
xvars = df.columns.tolist()
n_bootstrap = 10
sample_frac = 0.8
alpha = 0.05
stage1_method = "all"

edge_counts = defaultdict(int)

for i in range(n_bootstrap):
    df_sample = df.sample(frac=sample_frac, replace=True, random_state=i)
    test = Chi2RankTest(df_sample.to_numpy())
    input_parameters = {
        "ranktest_method": test,
        "citest_method": None,
        "stage1_method": stage1_method,
        "alpha_dict": {0: alpha, 1: alpha, 2: alpha, 3: alpha},
        "stage1_ges_sparsity": 2,
        "stage1_partition_thres": 3,
    }
    dotgraph, *_ = RLCD(sample=True, xvars=xvars, df_x=df_sample, input_parameters=input_parameters)
    
    for edge in dotgraph.edges():
        sorted_edge = tuple(sorted(edge))  # Treat A->B and B->A as same
        edge_counts[sorted_edge] += 1

# === Output ===
output_path = "bootstrap_results"
os.makedirs(output_path, exist_ok=True)

edge_freqs = {f"{a}--{b}": count / n_bootstrap for (a, b), count in edge_counts.items()}

with open(os.path.join(output_path, "edge_frequencies.json"), "w") as f:
    json.dump(edge_freqs, f, indent=4)

print("Edge frequency saved to:", os.path.join(output_path, "edge_frequencies.json"))
