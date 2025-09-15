import argparse
import pandas as pd
import numpy as np
import sys
import os

"""
Also outputs CPDAG
Adj[i,j]==-1 and Adj[j,i]==1 indicate i -> j; 
(Adj[i,j]==1 and Adj[j,i]==1) or Adj[i,j]==-1 and Adj[j,i]==-1 indicate i - j
"""

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'scm'))
# necessary imports for RLCD
from simulation_scenarios import *
from StructureLearning.RLCD.RLCD_alg import RLCD
from utils.Chi2RankTest import Chi2RankTest
from DGM.DataModel import DataModel

def run_causal_discovery_on_data(input_path, sample = 1, alpha = 0.01, rank_test_N_scaling = 1, stage1_method = "all", stage1_ges_sparsity = 2, stage1_partition_thres = 3):
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
    try:
        df_x = pd.read_csv(input_path)
        print(f"Data successfully loaded from {input_path}")
    except FileNotFoundError:
        sys.stderr.write(f"Error: The input file {input_path} was not found.\n")
        sys.exit(1)
    
    if sample:
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
    else:
        print(f"Need true DGM.")
        return None, None
    
    return estimated_adj_numpy, all_vars

if __name__ == "__main__":
    # The script uses argparse to get input and output paths from the command line
    parser = argparse.ArgumentParser(description="Causal Discovery Worker Script")
    parser.add_argument("input_path", type=str, help="Path to the input data CSV file.")
    parser.add_argument("output_path", type=str, help="Path to save the output adjacency matrix CSV.")
    parser.add_argument("alpha", type=float, help="The alpha level for the statistical tests.")
    args = parser.parse_args()

    # Get the estimated adjacency matrix and variable names
    # Corrected function call using keyword arguments
    estimated_adj_numpy, all_vars = run_causal_discovery_on_data(
        input_path=args.input_path,
        alpha=args.alpha
    )

    # Convert to DataFrame with names and save it to the specified output path
    estimated_adj_df = pd.DataFrame(estimated_adj_numpy, columns=all_vars, index= all_vars)
    # print(estimated_adj_df)
    estimated_adj_df.to_csv(args.output_path, index= True)

    print(f"Results successfully saved to {args.output_path}")