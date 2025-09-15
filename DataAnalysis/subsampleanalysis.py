import os, sys, re
import pandas as pd
from pathlib import Path
from SimpleUtils import count_three_types, merge_counts, counts_to_df

# --- paths & imports ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from convert_to_CPDAG import signed_to_CPDAG

# --- canonical vars ---
main_predCPDAG = pd.read_csv(f"{SCRIPT_DIR}/MainResults/worry_sleep_mistrustB_CPDAG.csv", index_col=0)
vars = list(main_predCPDAG.columns)
K_to_keep = len(vars)
Observed_to_keep = K_to_keep - sum(isinstance(c, str) and c.startswith("L") for c in vars)
print(K_to_keep); print(Observed_to_keep)

# --- inputs & summary file ---
IN_DIR = Path(SCRIPT_DIR) / "Subsampling"
files = sorted(IN_DIR.glob("Subsample_*.csv"))
curr_sum = pd.read_csv(f"{SCRIPT_DIR}/Subsampling/sum.csv")

if "Run_ID" not in curr_sum.columns:
    raise KeyError("sum.csv must contain a 'Run_ID' column.")
if "n_latents" not in curr_sum.columns:
    curr_sum["n_latents"] = pd.NA

dict_list = []

for f in files:
    # parse Run_ID from filename e.g., Bootstrap_123.csv -> "123"
    m = re.search(r"Subsample_(\d+)\.csv$", f.name)
    if not m:
        print(f"Skipping {f.name}: cannot parse Run_ID")
        continue
    run_id = m.group(1)  # keep as string for robust comparison

    # read bootstrap matrix
    bootstrap_df = pd.read_csv(f, index_col=0)
    bs_vars = list(bootstrap_df.columns)

    # sanity check for observed variables
    assert bs_vars[:Observed_to_keep] == vars[:Observed_to_keep], "Order mismatch vs. canonical vars"

    # count latents and update sum.csv
    n_latents = sum(isinstance(c, str) and c.startswith("L") for c in bs_vars)
    mask = (curr_sum["Run_ID"].astype(str) == run_id)
    if mask.any():
        curr_sum.loc[mask, "n_latents"] = n_latents
    else:
        print(f"Warning: Run_ID {run_id} not found in sum.csv")

    # CPDAG + counts (no alignment needed per your setup)
    bs_CPDAG = signed_to_CPDAG(bootstrap_df.to_numpy(), bs_vars)
    curr_dict = count_three_types(bs_CPDAG, vars, K_to_keep)
    dict_list.append(curr_dict)

# persist updated n_latents
curr_sum.to_csv(f"{SCRIPT_DIR}/Subsampling/sum_new.csv")

# merge & summarize
merged_counts = merge_counts(dict_list)
df_summary = counts_to_df(merged_counts, vars)
df_summary.to_csv(f"{SCRIPT_DIR}/Subsampling/summary_prop.csv")
