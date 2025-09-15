import os, sys 
from AnalysisUtils import toGraph
import pandas as pd

add = f"{os.getcwd()}"
sys.path.append(add)
from convert_to_CPDAG import signed_to_CPDAG

producedDAG = pd.read_csv(f"{add}/DataAnalysis/MainResults/worry_sleep_mistrustB.csv", index_col= 0)
vars = list(producedDAG.columns)
producedCPDAG = signed_to_CPDAG(producedDAG.to_numpy(), vars)
CPDAGdf = pd.DataFrame(producedCPDAG, index= vars, columns= vars)
CPDAGdf.to_csv(f"{add}/DataAnalysis/MainResults/worry_sleep_mistrustB_CPDAG.csv")
toGraph(producedCPDAG, "ordinalCPDAG", vars, f"{add}/DataAnalysis/MainResults")
