import pandas as pd
import pydot
import os
import sys
from AnalysisUtils import run_RLCD_on_data, toGraph
standard_path = os.path.dirname(os.path.abspath(__file__))
# print(os.getcwd())
based_input_path = f'{os.getcwd()}/data'
summed_all = f"{based_input_path}/summed_all.csv"
summed_demo = f"{based_input_path}/summed_demo.csv"
summed_df = pd.read_csv(summed_all)
demo_df = pd.read_csv(summed_demo)
original_all = f"{based_input_path}/original_data.csv"
original_df = pd.read_csv(original_all)

# --- paths & imports ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from convert_to_CPDAG import signed_to_CPDAG

df = demo_df.drop(columns=["Age", "household_income", "defences2", "Socialsupport"])

added_vars = ["MistrustC", "EverydayAnxiety", "SocialAnxiety", "Reasoning"]
df_added = summed_df[added_vars]       
new_df = pd.concat([df, df_added], axis=1)  # align by index
new_df = new_df.rename(columns={"RGPTSB": "MistrustB", 
                                "GPTSPARTA": "MistrustA"})
np_mistrust, all_vars = run_RLCD_on_data(new_df, sample=1, alpha= 0.001)

general_CPDAG = signed_to_CPDAG(np_mistrust, all_vars)
toGraph(general_CPDAG, "general_summed", all_vars)
df_CPDAG = pd.DataFrame(general_CPDAG, index= all_vars, columns= all_vars)
df_CPDAG.to_csv(f"{SCRIPT_DIR}/MainResults/general_CPDAG.csv")
