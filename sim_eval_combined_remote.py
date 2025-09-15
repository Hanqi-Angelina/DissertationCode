import os
import pandas as pd
# import numpy as np
import subprocess
import sys
import shutil  # for finding timeout/gtimeout
import csv

# ---- partial-results helpers ----
import csv, os

RESULT_FIELDS = ["Scenario", "Method", "Sample_Size", "Run_ID", "F1 Score", "SHD"]

def append_result_row(row, results_path):
    """Append one result dict to CSV immediately; write header on first write."""
    first = not os.path.exists(results_path)
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if first:
            w.writeheader()
        w.writerow(row)
        f.flush()

def load_done_keys(results_path):
    """Load finished (Scenario, Method, Sample_Size, Run_ID) from an existing CSV, if any."""
    if not os.path.exists(results_path):
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(results_path, usecols=["Scenario","Method","Sample_Size","Run_ID"])
        return set((r[0], r[1], int(r[2]), int(r[3])) for r in df.values.tolist())
    except Exception:
        done = set()
        with open(results_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    done.add((row["Scenario"], row["Method"], int(row["Sample_Size"]), int(row["Run_ID"])))
                except Exception:
                    pass
        return done
# --------------------------------------------------------

# Absolute paths & node-local scratch (if available)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.environ.get("SLURM_TMPDIR", os.getcwd())
os.makedirs(WORK_DIR, exist_ok=True)

# --- Imports ---
from metrics_revised import get_shd_f1
from simulation_scenarios import *
from GraphDrawer import *
from convert_to_CPDAG import adj_to_dataframe

def data_generator(scenario_classes, scenario_name, run_idx, sample_size, noise_type):
    dgm = scenario_classes[scenario_name](seed=run_idx)
    df_x, _ = dgm.generate_data(N= sample_size, normalized=True, noise_type = noise_type)
    true_adj_df = adj_to_dataframe(dgm)
    n_x = len(dgm.xvars)
    return df_x, true_adj_df, n_x

def main_orchestrator():

    num_runs = 80
    sample_sizes = [2000, 5000, 10000]
    methods = {
        "RLCD": "run_rlcd_worker.py",
        "LaHME": "run_lahme_worker.py"
    }
    conda_envs = {
        "RLCD": "scmenv1",
        "LaHME": "lahme2"
    }

    # Method-specific alpha levels
    alpha_levels = {
        "RLCD": 0.01,
        "LaHME": 0.01
    }
    scenario_classes = {
        "Measured1": Measured1, 
        "Measured2": Measured2, 
        "Simple_Case_Tree": Simple_Case_Tree,
        "Tree_Case_Flexible": Tree_Case_Flexible, 
        "Latent_Hierarchical_Structure": Latent_Hierarchical_Structure, 
        "LLCM_Flexible": LLCM_Flexible,
    }

    noise_type = 'gaussian'

    results = []
    start_id = 1

    # summary file path (used by append helper & for optional final write)
    final_output_path = os.path.join(os.getcwd(), f'evaluation_summary_{start_id}to{start_id+num_runs}_{noise_type}_new.csv')
    done = load_done_keys(final_output_path)

    print("--- Starting Consolidated Evaluation ---", flush=True)
    
    for run_idx in range(num_runs):
        for sample_size in sample_sizes:
            for scenario_name in scenario_classes.keys():
                print(f"  -> Generating data for {scenario_name} (N={sample_size}, Run={run_idx+start_id})", flush=True)

                df_x, true_adj_df, n_x = data_generator(
                    scenario_classes=scenario_classes, scenario_name=scenario_name, 
                    run_idx=run_idx+start_id, sample_size=sample_size, noise_type=noise_type
                )
                
                # 2. Loop through all methods
                for method_name, worker_script in methods.items():
                    key = (scenario_name, method_name, sample_size, run_idx + start_id)
                    if key in done:
                        print(f"    -> Skipping {method_name}: already in summary (resume mode).", flush=True)
                        continue

                    print(f"    -> Evaluating method: {method_name}", flush=True)

                    try:
                        # Pass data to the worker script via temporary files
                        temp_csv_input  = os.path.join(WORK_DIR, f"temp_input_{method_name}_{run_idx+start_id}_{sample_size}_{noise_type}.csv")
                        temp_csv_output = os.path.join(WORK_DIR, f"temp_output_{method_name}_{run_idx+start_id}_{sample_size}_{noise_type}.csv")
                        df_x.to_csv(temp_csv_input, index=False)
                        
                        current_alpha = alpha_levels[method_name]

                        # Build the command for subprocess
                        python_executable = ['conda', 'run', '-n', conda_envs[method_name], 'python', '-u']
                        worker_path = os.path.join(SCRIPT_DIR, worker_script)
                        command = [*python_executable, worker_path, temp_csv_input, temp_csv_output, str(current_alpha)]

                        # 5-minute cap if GNU timeout/gtimeout is available
                        runner = shutil.which("timeout") or shutil.which("gtimeout")
                        wrapped_command = [runner, "--signal=TERM", "--kill-after=5s", "300s", *command] if runner else command

                        # Run the external worker
                        result = subprocess.run(wrapped_command, capture_output=True, text=True, check=True)
                        
                        # Load the result and clean up
                        estimated_adj_df = pd.read_csv(temp_csv_output, index_col=0)
                        if os.path.exists(temp_csv_input):  os.remove(temp_csv_input)
                        if os.path.exists(temp_csv_output): os.remove(temp_csv_output)
                        
                        # 3. Metrics
                        shd_val, f1, _ = get_shd_f1(true_adj_df, estimated_adj_df, n_x, method=method_name)
                        row = {
                            'Scenario': scenario_name,
                            'Method': method_name,
                            'Sample_Size': sample_size,
                            'Run_ID': run_idx + start_id,
                            'F1 Score': f1,
                            'SHD': shd_val
                        }
                        # append immediately to summary CSV
                        append_result_row(row, final_output_path)
                        done.add(key)
                        results.append(row)

                        print(f"      -> F1 Score: {f1:.4f}, SHD: {shd_val}", flush=True)

                    except subprocess.CalledProcessError as e:
                        if os.path.exists(temp_csv_input): os.remove(temp_csv_input)
                        if e.returncode == 124:
                            print(f"      -> Skipping. {method_name} TIMED OUT (5 min).", flush=True)
                        else:
                            print(f"      -> Skipping. Worker script failed for {method_name}.", flush=True)
                        print(f"         Stdout: {e.stdout}", flush=True)
                        print(f"         Stderr: {e.stderr}", flush=True)
                        continue
                    except Exception as e:
                        if os.path.exists(temp_csv_input): os.remove(temp_csv_input)
                        print(f"      -> Skipping. An error occurred for {method_name}: {e}", flush=True)
                        continue

    # Optional final write:
    # If a summary file already exists (we've been appending), don't overwrite it.
    if results and not os.path.exists(final_output_path):
        pd.DataFrame(results).to_csv(final_output_path, index=False)

    print("\n--- Final Evaluation Complete ---", flush=True)
    print(f"Summary at: {final_output_path}", flush=True)

if __name__ == "__main__":
    main_orchestrator()
