import pandas as pd
import json
import argparse
from utils.logger import LOGGER
from scenarios import dgm_by_scenario
from utils.GraphDrawer import printGraph
import numpy as np
from StructureLearning.RLCD.RLCD_alg import RLCD
from utils.OraclePartialCorrTest import OraclePartialCorrTest
from utils.OracleRankTest import OracleRankTest
from utils.Chi2RankTest import Chi2RankTest
import os
from scenarios import dgm_by_scenario
import pydot

if __name__ == "__main__":

    parser = argparse.ArgumentParser("main")
    parser.add_argument("--s", type=str, default="multitasking")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--stage1_ges_sparsity", type=float, default=2) # sparsity for fges (a bit sparse?)
    parser.add_argument("--stage1_partition_thres", type=float, default=3) # 3 default (fewer vs more partitions)
    parser.add_argument("--stage1_method", type=str, default="fges") # default stage 1 methods is fges
    parser.add_argument("--alpha", type=float, default=0.01) 
    parser.add_argument("--rank_test_N_scaling", type=float, default=1) # default scaling is 1
    
    args = parser.parse_args()

    dgm = dgm_by_scenario[args.s]
    
    if args.sample:
        df_x, df_v = dgm.generate_data(N=args.n, normalized=True)
        input_parameters = {
            "ranktest_method": Chi2RankTest(df_x.to_numpy(), args.rank_test_N_scaling),
            "citest_method": None,
            "stage1_method": args.stage1_method,
            "alpha_dict": {0:args.alpha, 1:args.alpha, 2:args.alpha, 3:args.alpha},
            "stage1_ges_sparsity": args.stage1_ges_sparsity,
            "stage1_partition_thres": args.stage1_partition_thres
        }
        result_dotgraph, result_stage1_dotgraph, _, _ = RLCD(args.sample, dgm.xvars, df_x, input_parameters)
    else:
        input_parameters = {
            "ranktest_method": OracleRankTest(dgm),
            "citest_method": OraclePartialCorrTest(dgm, np.zeros((1, len(dgm.xvars)))),
            "stage1_method": args.stage1_method,
            "alpha_dict": {0:args.alpha, 1:args.alpha, 2:args.alpha, 3:args.alpha},
            "stage1_ges_sparsity": args.stage1_ges_sparsity,
            "stage1_partition_thres": args.stage1_partition_thres
        }
        result_dotgraph, result_stage1_dotgraph, _, _ = RLCD(args.sample, dgm.xvars, None, input_parameters)

    plots_save_path = f'{args.s}_results'
    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)

    if args.sample: 
        printGraph(result_dotgraph, f'{plots_save_path}/alpha{args.alpha}_rtscale{args.rank_test_N_scaling}_N{args.n}.png')


def run_rlcd_from_data(scenario_name, dgm_object, df_x, sample = 1, alpha = 0.01, rank_test_N_scaling = 1, stage1_method = "all", stage1_ges_sparsity = 2, stage1_partition_thres = 3):
    """
    Runs the RLCD algorithm on the given data and returns the adjacency matrix.
    
    Args:
        scenario_name: The name of the scenario (e.g., "LLCM_LLH_Case1").
        dgm_object: The DataModel or LinearSCM object for the scenario.
        df_x: The pandas DataFrame of observed data.
        sample: A boolean indicating whether to use sampling (non-oracle) or not.
        alpha: The alpha value for statistical tests.
        rank_test_N_scaling: The scaling factor for the rank test.
        stage1_method: The method for stage 1 (e.g., "fges").
        stage1_ges_sparsity: The sparsity for the FGES method.
        stage1_partition_thres: The threshold for partitioning.

    Returns:
        A NumPy adjacency matrix of the discovered graph.
    """
    
    # Use the scenario name to create a dedicated save path, mirroring your original script.
    plots_save_path = f'Simulation/{scenario_name}/results'
    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)
    
    if sample:
        input_parameters = {
            "ranktest_method": Chi2RankTest(df_x.to_numpy(), rank_test_N_scaling),
            "citest_method": None,
            "stage1_method": stage1_method,
            "alpha_dict": {0: alpha, 1: alpha, 2: alpha, 3: alpha},
            "stage1_ges_sparsity": stage1_ges_sparsity,
            "stage1_partition_thres": stage1_partition_thres
        }
        result_combined_dotgraph, result_stage1_dotgraph, Adj_combined, all_vars = RLCD(sample, dgm_object.xvars, df_x, input_parameters)
    else:
        # Note: This branch is for an oracle setting and doesn't use the observed data directly for discovery.
        input_parameters = {
            "ranktest_method": OracleRankTest(dgm_object),
            "citest_method": OraclePartialCorrTest(dgm_object, np.zeros((1, len(dgm_object.xvars)))),
            "stage1_method": stage1_method,
            "alpha_dict": {0: alpha, 1: alpha, 2: alpha, 3: alpha},
            "stage1_ges_sparsity": stage1_ges_sparsity,
            "stage1_partition_thres": stage1_partition_thres
        }
        result_combined_dotgraph, result_stage1_dotgraph, Adj_combined, all_vars = RLCD(sample, dgm_object.xvars, None, input_parameters)
    
    if sample: 
        printGraph(result_combined_dotgraph, f'{plots_save_path}/alpha{alpha}_rtscale{rank_test_N_scaling}.png')
    
    return result_combined_dotgraph, result_stage1_dotgraph, Adj_combined, all_vars