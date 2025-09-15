import os
import time
import csv
import signal
import subprocess

import pandas as pd
import numpy as np

# ---- partial-results helpers----
RESULT_FIELDS = ["Scenario", "Method", "Threshold", "Sample_Size", "Run_ID", "F1 Score", "SHD"]

def append_result_row(row, results_path):
    """
    Append one result dict to CSV immediately; 
    write header on first write.
    """
    first = not os.path.exists(results_path)
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if first:
            w.writeheader()
        w.writerow(row)
        f.flush()

def load_done_keys(results_path):
    """
    Load finished (Scenario, Method, Threshold, Sample_Size, Run_ID) keys
    from an existing CSV, if any.
    """
    if not os.path.exists(results_path):
        return set()
    try:
        df = pd.read_csv(
            results_path,
            usecols=["Scenario", "Method", "Threshold", "Sample_Size", "Run_ID"]
        )
        df["Scenario"] = df["Scenario"].astype(str)
        df["Method"] = df["Method"].astype(str)
        df["Threshold"] = df["Threshold"].astype(str)
        df["Sample_Size"] = df["Sample_Size"].astype("Int64").fillna(0).astype(int)
        df["Run_ID"] = df["Run_ID"].astype("Int64").fillna(0).astype(int)
        return set(
            (row.Scenario, row.Method, row.Threshold, int(row.Sample_Size), int(row.Run_ID))
            for row in df.itertuples(index=False)
        )
    except Exception:
        done = set()
        with open(results_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    done.add((
                        str(row["Scenario"]),
                        str(row["Method"]),
                        str(row["Threshold"]),
                        int(row["Sample_Size"]),
                        int(row["Run_ID"]),
                    ))
                except Exception:
                    pass
        return done
# --------------------------------------------------------

# Absolute paths & node-local scratch (if available)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.environ.get("SLURM_TMPDIR", os.getcwd())  # SLURM local scratch if present
os.makedirs(WORK_DIR, exist_ok=True)

# --- Your domain imports ---
from metrics_revised import get_shd_f1
from simulation_scenarios import *
from convert_to_CPDAG import adj_to_dataframe

def data_generator(scenario_classes, scenario_name, run_idx, sample_size, noise_type):
    dgm = scenario_classes[scenario_name](seed=run_idx)
    df_x, _ = dgm.generate_data(N=sample_size, normalized=True, noise_type=noise_type)
    true_adj_df = adj_to_dataframe(dgm)
    n_x = len(dgm.xvars)
    return df_x, true_adj_df, n_x

# fixed thresholds from literature
TAU = {
    "symmetric":  np.array([-1.5, -0.5, 0.5, 1.5]),
    "mild":       np.array([-0.05, 0.77, 1.34, 1.88]),
    "moderate":   np.array([0.67, 1.28, 1.645, 2.05]),
}

def thresholding(df, type, start_at=1):
    tau = TAU[type]
    return df.apply(lambda col: start_at + np.digitize(col, tau))

# -------- Robust, tree-killing timeout (300s) --------
# Use 8 CPUs per *serial* worker by default (pulls from SLURM if set)
DEFAULT_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
THREADS_PER_WORKER = str(DEFAULT_CPUS)  

SINGLE_WORKER_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", THREADS_PER_WORKER),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", THREADS_PER_WORKER),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", THREADS_PER_WORKER),
    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", THREADS_PER_WORKER),
}

def run_with_timeout(cmd, timeout_sec=300, cwd=None):
    """
    Launch cmd in its own process group; on timeout send SIGTERM then SIGKILL.
    Returns (rc, out, err). Uses rc=124 for timeout (GNU timeout convention).
    """
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=SINGLE_WORKER_ENV,
        start_new_session=True,  # new PGID to kill the whole tree
    )
    try:
        out, err = p.communicate(timeout=timeout_sec)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            out, err = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
        return 124, out, err
# -----------------------------------------------------

def main_orchestrator():
    print("SLURM_CPUS_PER_TASK=", os.getenv("SLURM_CPUS_PER_TASK"))
    print("Thread env:", SINGLE_WORKER_ENV["OMP_NUM_THREADS"], SINGLE_WORKER_ENV["MKL_NUM_THREADS"],
          SINGLE_WORKER_ENV["OPENBLAS_NUM_THREADS"], SINGLE_WORKER_ENV["NUMEXPR_NUM_THREADS"], flush=True)

    num_runs = 80

    arr = os.getenv("SLURM_ARRAY_TASK_ID")
    if arr:
        run_ids = [int(arr)] # group
    else:
        run_ids = list(range(1, num_runs+1)) # original
    
    sample_sizes = [10000]
    methods = {
        "RLCD": "run_rlcd_worker.py",
        "LaHME": "run_lahme_worker.py",
    }
    conda_envs = {
        "RLCD": "scmenv1",
        "LaHME": "lahme2",
    }
    alpha_levels = {
        "RLCD": 0.01,
        "LaHME": 0.01,
    }
    scenario_classes = {
        "Simple_Case_Tree": Simple_Case_Tree,
        "Latent_Hierarchical_Structure": Latent_Hierarchical_Structure,
        "Fanning": Fanning_Indicator,
    }
    threshold_types = ["symmetric", "mild", "moderate"]
    noise_type = 'gaussian'

    print(f"--- Starting (serial, {THREADS_PER_WORKER} threads/worker) ---", flush=True)

    for run_idx in run_ids:
        results = []
        ok = timeouts = fails = 0
        final_output_path = os.path.join(
            os.getcwd(),
            f"evaluation_summary_run{run_idx}_{noise_type}_thresholding.csv"
        )
        if (not os.path.exists(final_output_path)) or (os.path.getsize(final_output_path) == 0):
            with open(final_output_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()

        done = load_done_keys(final_output_path)

        for sample_size in sample_sizes:
            for scenario_name in scenario_classes.keys():
                for threshold_type in threshold_types:
                    print(f"  -> Generating data for {scenario_name} {threshold_type} (N={sample_size}, Run={run_idx})", flush=True)

                    df_x, true_adj_df, n_x = data_generator(
                        scenario_classes=scenario_classes, scenario_name=scenario_name,
                        run_idx=run_idx, sample_size=sample_size, noise_type=noise_type
                    )
                    df_x = thresholding(df_x, threshold_type)

                    for method_name, worker_script in methods.items():
                        key = (scenario_name, method_name, threshold_type, sample_size, run_idx)
                        if key in done:
                            print(f"    -> Skipping {method_name}: already in summary (resume mode).", flush=True)
                            continue

                        print(f"    -> Evaluating method: {method_name}", flush=True)

                        temp_csv_input  = os.path.join(WORK_DIR, f"temp_input_{scenario_name}_{method_name}_{run_idx}_{sample_size}_{threshold_type}.csv")
                        temp_csv_output = os.path.join(WORK_DIR, f"temp_output_{scenario_name}_{method_name}_{run_idx}_{sample_size}_{threshold_type}.csv")

                        try:
                            df_x.to_csv(temp_csv_input, index=False)
                            current_alpha = alpha_levels[method_name]

                            python_executable = ['conda', 'run', '-n', conda_envs[method_name], 'python', '-u']
                            worker_path = os.path.join(SCRIPT_DIR, worker_script)
                            command = [*python_executable, worker_path, temp_csv_input, temp_csv_output, str(current_alpha)]

                            # Serial run with hard 300s cap that kills the entire tree
                            t0 = time.time()
                            print(f"      -> START {method_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                            print("      -> CMD:", " ".join(command), flush=True)
                            rc, out, err = run_with_timeout(command, timeout_sec=300)
                            elapsed = time.time() - t0
                            print(f"      -> END   {method_name} rc={rc} elapsed={elapsed:.1f}s", flush=True)
                            
                            if rc != 0:
                                timeouts += (rc == 124)
                                fails += (rc != 124)
                                if os.path.exists(temp_csv_input): os.remove(temp_csv_input)
                                if rc == 124:
                                    print(f"      -> Skipping. {method_name} TIMED OUT (5 min).", flush=True)
                                elif rc < 0:
                                    print(f"      -> Skipping. {method_name} terminated by signal {-rc}.", flush=True)
                                else:
                                    print(f"      -> Skipping. Worker script failed for {method_name} (rc={rc}).", flush=True)
                                if out: print(f"         Stdout: {out}", flush=True)
                                if err: print(f"         Stderr: {err}", flush=True)
                                continue
                            else: 
                                ok += 1

                            # ensure output exists and is non-empty
                            if not (os.path.exists(temp_csv_output) and os.path.getsize(temp_csv_output) > 0):
                                if os.path.exists(temp_csv_input): os.remove(temp_csv_input)
                                print(f"      -> Skipping. {method_name} produced no output file.", flush=True)
                                continue

                            estimated_adj_df = pd.read_csv(temp_csv_output, index_col=0)
                            if os.path.exists(temp_csv_input):  os.remove(temp_csv_input)
                            if os.path.exists(temp_csv_output): os.remove(temp_csv_output)

                            # Metrics
                            shd_val, f1, _ = get_shd_f1(true_adj_df, estimated_adj_df, n_x, method=method_name)
                            row = {
                                'Scenario': scenario_name,
                                'Method': method_name,
                                'Threshold': threshold_type,
                                'Sample_Size': sample_size,
                                'Run_ID': run_idx,
                                'F1 Score': f1,
                                'SHD': shd_val
                            }
                            append_result_row(row, final_output_path)
                            done.add(key)
                            results.append(row)
                            print(f"      -> F1 Score: {f1:.4f}, SHD: {shd_val}", flush=True)

                        except Exception as e:
                            # always remove the input temp to avoid stale re-use
                            if os.path.exists(temp_csv_input): os.remove(temp_csv_input)
                            print(f"      -> Skipping. An error occurred for {method_name}: {e}", flush=True)
                            # keep output for post-mortem
                            continue
        print(f"[run {run_idx}] ok={ok} timeout={timeouts} fail={fails}", flush=True)
        print("\n--- Final Evaluation Complete ---", flush=True)
        print(f"Summary at: {final_output_path}", flush=True)

if __name__ == "__main__":
    main_orchestrator()
