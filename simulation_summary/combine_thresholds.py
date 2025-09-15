import os
import pandas as pd
from math import sqrt

# --- settings ---
seeds = list(range(1, 22)) + list(range(23, 42)) + list(range(43, 81))
# print(seeds)

KEYS = ["Scenario","Method","Threshold","Sample_Size","Run_ID"]
VALS = ["F1 Score","SHD"]
expected_per_run = 18       # 2 methods × 3 thresholds × 3 scenarios

# --- merge & de-duplicate ---
rows = []
status = []
base_dir = os.path.dirname(os.path.abspath(__file__))
threshold_dir = os.path.join(base_dir, "thresholds")
for seed in seeds:
    f = os.path.join(threshold_dir, f"evaluation_summary_run{seed}_gaussian_thresholding.csv")
    try:
        df = pd.read_csv(f, usecols=KEYS+VALS)
    except Exception:
        df = pd.DataFrame(columns=KEYS+VALS)
    # keep only this run's rows, unique keys
    status.append({"run": seed, "file": os.path.basename(f),
                    "rows": len(df)})
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEYS+VALS)
# print(sum(all_df.duplicated(subset=KEYS))) # checked, no duplicates

# write merged tables
all_df.to_csv("evaluation_summary_all_dedup.csv", index=False)

# per-group summary (mean, SD) for F1 and SHD
g = all_df.groupby(["Scenario","Method","Threshold","Sample_Size"], as_index=False)
sum_df = g.agg(
    n=("Run_ID","nunique"),
    F1_mean=("F1 Score","mean"),
    F1_sd=("F1 Score","std"),
    SHD_mean=("SHD","mean"),
    SHD_sd=("SHD","std"),
)
sum_df["F1_se"]  = sum_df["F1_sd"]  / sum_df["n"].apply(sqrt)
sum_df["SHD_se"] = sum_df["SHD_sd"] / sum_df["n"].apply(sqrt)
sum_df.to_csv("summary_group_stats.csv", index=False)