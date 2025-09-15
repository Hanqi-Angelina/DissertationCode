import argparse
import pandas as pd
import numpy as np
import sys
import os

"""
Adj[i,j] = 1 means j -> i
"""


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
# This assumes the lahme folder is in the same directory as the script.
sys.path.append(os.path.join(script_dir, 'lahme'))
# necessary imports for LaHME
from Main_LiNGLaH_LaHME import LaHME

def convert_to_signed(m):
    """
    This is for converting the binary DAG case to the general signed case.
    """
    conventional = m - m.T
    return conventional

def run_lahme(input_path, alpha=0.01):
    """
    Runs the LaHME algorithm on the given data and returns the estimated adjacency matrix.
    """
    try:
        df_x = pd.read_csv(input_path)
        print(f"Data successfully loaded from {input_path}")
    except FileNotFoundError:
        sys.stderr.write(f"Error: The input file {input_path} was not found.\n")
        sys.exit(1)
    
    # LaHME returns a pandas DataFrame, which is the estimated adjacency matrix
    estimated_adj_df = LaHME(df_x, alpha)

    # convert to the general signed matrix here
    return convert_to_signed(estimated_adj_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaHME Worker Script")
    parser.add_argument("input_path", type=str, help="Path to the input data CSV file.")
    parser.add_argument("output_path", type=str, help="Path to save the output adjacency matrix CSV.")
    parser.add_argument("alpha", type=float, help="The alpha level for the statistical tests.")
    args = parser.parse_args()

    # Get the estimated adjacency matrix (as a DataFrame)
    estimated_adj_df = run_lahme(
        input_path=args.input_path,
        alpha=args.alpha
    )

    # Save the DataFrame to the specified output path, only column names needed
    estimated_adj_df.to_csv(args.output_path, index=True)

    print(f"Results successfully saved to {args.output_path}")