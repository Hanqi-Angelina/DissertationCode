import numpy as np
import os
import pandas as pd
import csv
import sys

# ----- PATH FIX -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from metrics_revised import get_shd_f1
from simulation_scenarios import *
from AnalysisUtils import run_RLCD_on_data, get_col_dict, subsample_indices

# Absolute paths & node-local scratch (if available)
WORK_DIR = os.environ.get("SLURM_TMPDIR", os.getcwd())
os.makedirs(WORK_DIR, exist_ok=True)

save_folder = f'{SCRIPT_DIR}/Subsampling'
os.makedirs(save_folder, exist_ok=True)

# Read data from project-root/data (instead of cwd/data)
based_input_path = f'{PROJECT_ROOT}/data'
original_all = f"{based_input_path}/original_data.csv"

# Stage to SLURM local scratch if present
slurm_tmp = os.environ.get("SLURM_TMPDIR")
if slurm_tmp:
    import shutil
    local_original = os.path.join(WORK_DIR, "original_data.csv")
    if not os.path.exists(local_original):
        shutil.copy2(original_all, local_original)
    read_path = local_original
else:
    read_path = original_all

original_df = pd.read_csv(read_path)

DEFAULT_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
THREADS_PER_WORKER = str(DEFAULT_CPUS)

SINGLE_WORKER_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", THREADS_PER_WORKER),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", THREADS_PER_WORKER),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", THREADS_PER_WORKER),
    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", THREADS_PER_WORKER),
}

# Record SHD
RESULT_FIELDS = ["Run_ID", "SHD", "Error"]

# Pre-create header once (so no duplicate headers in arrays)
summary_path = f'{save_folder}/sum.csv'
if not os.path.exists(summary_path):
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Run_ID,SHD,Error\n")

# ----- ATOMIC APPEND FIX -----
def append_result_row(row, results_path):
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    import io, os as _os
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=RESULT_FIELDS)
    w.writerow(row)
    data = buf.getvalue().encode("utf-8")
    fd = _os.open(results_path, _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND, 0o644)
    try:
        _os.write(fd, data)
    finally:
        _os.close(fd)

# ----- TIMEOUT HANDLER (moved here so deps exist) -----
import signal
_current_run_id = None
_wrote_timeout = False

def _graceful_timeout(signum, frame):
    global _wrote_timeout
    if _wrote_timeout:
        return
    _wrote_timeout = True
    rid = _current_run_id if _current_run_id is not None else os.getenv("SLURM_ARRAY_TASK_ID", "NA")
    row = {"Run_ID": int(rid) if str(rid).isdigit() else rid,
           "SHD": "",
           "Error": "TIMEOUT: received SIGTERM before SLURM kill"}
    try:
        append_result_row(row, summary_path)
    finally:
        sys.exit(0)

signal.signal(signal.SIGTERM, _graceful_timeout)
signal.signal(signal.SIGINT, _graceful_timeout)

num_runs = 1
arr = os.getenv("SLURM_ARRAY_TASK_ID")
if arr:
    run_ids = [int(arr)]
else:
    run_ids = list(range(1, num_runs+1))

desired_columns = ['Worry','MistrustB', 'Sleep']
grouped = get_col_dict(original_df)
subset_columns = [x for k, v in grouped.items() if k in desired_columns for x in v]
predicted_df = pd.read_csv(f"{SCRIPT_DIR}/MainResults/worry_sleep_mistrustB.csv", index_col=0)

df_new = original_df[subset_columns]
df_std = (df_new - df_new.mean()) / df_new.std()
n = df_std.shape[0]
n_x = df_std.shape[1]

for run_id in run_ids:
    _current_run_id = run_id
    new_ids = subsample_indices(n=n, seed=run_id, frac= 0.5)
    new_sample = df_std.iloc[new_ids]
    print(new_sample)

    try:
        adj_mat, all_vars = run_RLCD_on_data(new_sample, 1, alpha=0.001)
        estimated_adj_df = pd.DataFrame(adj_mat, columns=all_vars, index=all_vars)

        shd, _, best_perm_df = get_shd_f1(
            predicted_df, estimated_adj_df, n_x=n_x, method="RLCD"
        )

        pd.DataFrame(best_perm_df).to_csv(f'{save_folder}/Subsample_{run_id}.csv', index=True)
        row = {'Run_ID': run_id, 'SHD': shd, 'Error': ''}
        append_result_row(row, summary_path)

    except Exception as e:
        row = {'Run_ID': run_id, 'SHD': '', 'Error': f'{type(e).__name__}: {e}'}
        append_result_row(row, summary_path)
