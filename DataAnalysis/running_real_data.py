import numpy as np
import pandas as pd
from itertools import combinations
import itertools
import networkx as nx
import random, os, sys
from AnalysisUtils import run_RLCD_on_data, toGraph, get_col_dict

based_input_path = f'{os.getcwd()}/data'
original_all = f"{based_input_path}/original_data.csv"
original_df = pd.read_csv(original_all)

standard_path = standard_path = os.path.dirname(os.path.abspath(__file__))
grouped = get_col_dict(original_df)
print(grouped)

desired_columns = ['Worry','MistrustB', 'Sleep']

subset_columns = [x for k, v in grouped.items() if k in desired_columns for x in v]

df_new = original_df[subset_columns]
df_std = (df_new - df_new.mean()) / df_new.std()

np1, npvars = run_RLCD_on_data(df_x=df_std, sample=1, alpha=0.001)
toGraph(np1,'worry_sleep_mistrustB_new', npvars, f"{standard_path}/MainResults")
df_predicted = pd.DataFrame(np1, index=npvars, columns=npvars)
df_predicted.to_csv(f'{standard_path}/MainResults/worry_sleep_mistrustB_new.csv')


