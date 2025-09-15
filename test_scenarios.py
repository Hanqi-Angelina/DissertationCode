from simulation_scenarios import * 
from convert_to_CPDAG import adj_to_dataframe, toGraph
"""
Get all the true DAGs here. 
"""

dgm1 = Measured1(seed = 0)
measured1_true = adj_to_dataframe(dgm1)
toGraph(measured1_true.to_numpy(), "measured1", list(measured1_true.columns))

dgm2 = Measured2(seed = 0)
measured2_true = adj_to_dataframe(dgm2)
toGraph(measured2_true.to_numpy(), "measured2", list(measured2_true.columns))

dgm3 = Simple_Case_Tree(seed = 0)
measured3_true = adj_to_dataframe(dgm3)
toGraph(measured3_true.to_numpy(), "simple_tree", list(measured3_true.columns))

dgm4 = Tree_Case_Flexible(seed = 0)
measured4_true = adj_to_dataframe(dgm4)
toGraph(measured4_true.to_numpy(), "flexible_tree", list(measured4_true.columns))

dgm5 = Latent_Hierarchical_Structure(seed = 0)
measured5_true = adj_to_dataframe(dgm5)
toGraph(measured5_true.to_numpy(), "LHM", list(measured5_true.columns))

dgm6 = LLCM_Flexible(seed = 0)
measured6_true = adj_to_dataframe(dgm6)
toGraph(measured6_true.to_numpy(), "LLCM", list(measured6_true.columns))

dgm7 = Fanning_Indicator(seed= 0)
dgm7_true = adj_to_dataframe(dgm7)
toGraph(dgm7_true.to_numpy(), "Fanning", list(dgm7_true.columns))