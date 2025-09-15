import pandas as pd
import pydot
import os
import sys
from collections import defaultdict
import numpy as np

# --- PATHS (robust local & SLURM) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# import from project root (one level up)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# if scm/ is a sibling of DataAnalysis/, make it importable too
SCM_DIR = os.path.join(PROJECT_ROOT, "scm")
if os.path.isdir(SCM_DIR) and SCM_DIR not in sys.path:
    sys.path.append(SCM_DIR)

print("SCRIPT_DIR     :", SCRIPT_DIR)
print("PROJECT_ROOT   :", PROJECT_ROOT)
print("SCM_DIR exists :", os.path.isdir(SCM_DIR))
print(SCM_DIR)

from GraphDrawer import AdjToGraph

standard_path = SCRIPT_DIR  # unchanged intent; lives beside this script
def toGraph(M, name, var_names, base_folder=f"{standard_path}/DataResults"):
    AdjToGraph(M, var_names).toDot(f'{base_folder}/{name}.dot')
    g, = pydot.graph_from_dot_file(f'{base_folder}/{name}.dot')
    g.write_png(f'{base_folder}/{name}.png', prog='dot')

from StructureLearning.RLCD.RLCD_alg import RLCD
from utils.Chi2RankTest import Chi2RankTest
from DGM.DataModel import DataModel

def run_RLCD_on_data(df_x, sample = 1, alpha = 0.01, rank_test_N_scaling = 1, stage1_method = "all", stage1_ges_sparsity = 2, stage1_partition_thres = 3):
    """
    Runs the RLCD algorithm on the given data and returns the adjacency matrix.
    
    Args:
        input_path: Path to the input data CSV file.
        # output_path is removed from here
        sample: A boolean indicating whether to use sampling (non-oracle) or not.
        alpha: The alpha value for statistical tests.
        ... (other parameters) ...

    Returns:
        A tuple of (estimated_adj_numpy, all_vars)
    """
    
    if sample:
        xvars = list(df_x.columns)        
        df_x  = df_x[xvars].copy()      
        dgm_object = DataModel(df_x, df_x) 

        input_parameters = {
            "ranktest_method": Chi2RankTest(df_x.to_numpy(), rank_test_N_scaling),
            "citest_method": None,
            "stage1_method": stage1_method,
            "alpha_dict": {0: alpha, 1: alpha, 2: alpha, 3: alpha},
            "stage1_ges_sparsity": stage1_ges_sparsity,
            "stage1_partition_thres": stage1_partition_thres
        }
        _, _, estimated_adj_numpy, all_vars = RLCD(sample, dgm_object.xvars, df_x, input_parameters)
        # print(xvars == dgm_object.xvars)
    else:
        print(f"Need true DGM.")
        return None, None
    
    return estimated_adj_numpy, all_vars

def get_col_dict(df):
    grouped = defaultdict(list)
    for col in df.columns:
        prefix = col.split("_", 1)[0]  # take everything before the first underscore
        grouped[prefix].append(col)
    return grouped

def bootstrap_indices(n, seed):
    """
    Return indices for one bootstrap sample (with replacement).
    n: total number of rows
    seed: seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    size = n
    return rng.integers(low=0, high=n, size=size, endpoint=False)

def subsample_indices(n,seed,frac=0.5):
    """
    Return indices for one subsample (without replacement).
    n: total number of rows
    seed: seed for reproducibility
    frac: fraction of rows to keep (e.g., 0.5 for n/2)
    """
    rng = np.random.default_rng(seed=seed)
    k = max(1, int(round(frac * n)))
    return rng.choice(n, size=k, replace=False) 