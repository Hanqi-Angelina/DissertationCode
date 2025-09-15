from LinearSCM import LinearSCM
"""
Develop the seven cases of linear SCMs. 
"""
def Measured1(seed = 0): 
    """
    The observed-only model with a connected graph. 
    """
    dgm = LinearSCM(seed = seed)
    n_obs = 12
    for j in range(1, n_obs + 1):
        dgm.add_variable(f"X{j}", True)  # Add observed variables
    dgm.add_edge("X1", "X2")
    dgm.add_edge("X1", "X3")
    dgm.add_edge("X1", "X4")
    dgm.add_edge("X1", "X5")
    dgm.add_edge("X2", "X5")
    dgm.add_edge("X2", "X6")
    dgm.add_edge("X2", "X7")
    dgm.add_edge("X3", "X8")
    dgm.add_edge("X3", "X9")
    dgm.add_edge("X3", "X10")
    dgm.add_edge("X4", "X11")
    dgm.add_edge("X3", "X11")
    dgm.add_edge("X4", "X12")

    return dgm

def Measured2(seed = 0):
    """
    An observed-only model with two clusters. 
    """
    dgm = LinearSCM(seed = seed)
    n_obs = 12
    for j in range(1, n_obs + 1):
        dgm.add_variable(f"X{j}", True)  # Add observed variables
    # add a triangle structure first
    dgm.add_edge("X1", "X2")
    dgm.add_edge("X1", "X3")
    dgm.add_edge("X2", "X3")   
    dgm.add_edge("X3", "X4")
    dgm.add_edge("X3", "X5")
    dgm.add_edge("X2", "X6")
    # three parents to the same children (some structure do not handle this)
    dgm.add_edge("X8", "X7")
    dgm.add_edge("X9", "X7")
    dgm.add_edge("X10", "X7")
    dgm.add_edge("X7", "X11")
    dgm.add_edge("X7", "X12")
    
    return dgm

def Simple_Case_Tree(seed = 0):
    """
    Simple two cluster graphs. Latent Hierarchical. 
    Check that if two observed are unrelated, there will be no edges.
    Only latents cause measured variables. 
    """
    n_lats = 3;  n_obs = 12
    dgm = LinearSCM(seed = seed)
    for i in range(1, n_lats + 1):
        dgm.add_variable(f"L{i}", False) # Add latent variables
    for j in range(1, n_obs + 1):
        dgm.add_variable(f"X{j}", True)  # Add observed variables
    dgm.add_edge("L1", "X1")
    dgm.add_edge("L1", "X2")
    dgm.add_edge("L1", "X3")
    dgm.add_edge("L1", "X4")
    dgm.add_edge("L1", "X5")
    dgm.add_edge("L2", "X7")
    dgm.add_edge("L2", "X8")
    dgm.add_edge("L2", "X9")
    dgm.add_edge("L2", "X10")
    dgm.add_edge("L2", "X11")

    dgm.add_edge("L3", "L1")
    dgm.add_edge("L3", "L2")

    return dgm

def Tree_Case_Flexible(seed=0):
    """
    Simple tree case where observed can cause latents. Revised from Dong. 
    """
    dgm = LinearSCM(seed=seed)
    dgm.add_variable("X1", True)
    dgm.add_variable("L1", False)
    dgm.add_variable("X2", True)
    dgm.add_variable("L2", False)

    dgm.add_edge("X1", "L1")
    dgm.add_edge("X1", "X2")
    dgm.add_edge("X1", "L2")

    dgm.add_variable("X3", True)
    dgm.add_variable("L3", False)
    dgm.add_variable("X4", True)

    dgm.add_edge("L1", "X3")
    dgm.add_edge("L1", "L3")
    dgm.add_edge("L1", "X4")

    dgm.add_variable("X5", True)
    dgm.add_variable("X6", True)

    dgm.add_edge("X2", "X5")
    dgm.add_edge("X2", "X6")

    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)
    dgm.add_variable("X9", True)

    dgm.add_edge("L2", "X7")
    dgm.add_edge("L2", "X8")
    dgm.add_edge("L2", "X9")

    dgm.add_variable("X10", True)
    dgm.add_variable("X11", True)
    dgm.add_variable("X12", True)

    dgm.add_edge("L3", "X10")
    dgm.add_edge("L3", "X11")
    dgm.add_edge("L3", "X12")

    return dgm

def Latent_Hierarchical_Structure(seed = 0): 
    """
    Latent Hierarchical Structure other than Trees. Assume that Latent Cause Observed Only.
    """
    dgm = LinearSCM(seed=seed)
    n_lats = 5;  n_obs = 12
    for i in range(1, n_lats + 1):
        dgm.add_variable(f"L{i}", False) # Add latent variables
    for j in range(1, n_obs + 1):
        dgm.add_variable(f"X{j}", True)  # Add observed variables
    dgm.add_edge("L1", "L3")
    dgm.add_edge("L2", "L3")
    dgm.add_edge("L2", "L1")
    dgm.add_edge("L2", "L5")  
    dgm.add_edge("L3", "L4")
    dgm.add_edge("L3", "L5")
    dgm.add_edge("L3", "X3")
    dgm.add_edge("L1", "X1")
    dgm.add_edge("L1", "X2")
    dgm.add_edge("L4", "X4")
    dgm.add_edge("L3", "X5")
    dgm.add_edge("L4", "X6")
    dgm.add_edge("L4", "X10")
    dgm.add_edge("L2", "X7")
    dgm.add_edge("L5", "X8") 
    dgm.add_edge("L3", "X9")
    dgm.add_edge("L5", "X11")
    dgm.add_edge("L5", "X12") 

    return dgm

def LLCM_Flexible(seed=0):
    """
    A more general graph with measured cause latents exist. 
    """
    dgm = LinearSCM(seed=seed)
    dgm.add_variable("L1", False)
    dgm.add_variable("L2", False)
    dgm.add_variable("L3", False)
    dgm.add_variable("L4", False)

    dgm.add_variable("X1", True)
    dgm.add_variable("X2", True)
    dgm.add_variable("X3", True)
    dgm.add_variable("X4", True)
    dgm.add_variable("X5", True)
    dgm.add_variable("X6", True)
    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)
    dgm.add_variable("X9", True)
    dgm.add_variable("X10", True)
    dgm.add_variable("X11", True)
    dgm.add_variable("X12", True)

    dgm.add_edge("L1", "X1")
    dgm.add_edge("X1", "L2")
    dgm.add_edge("L1", "L3")
    dgm.add_edge("X1", "L3")
    dgm.add_edge("X1", "X3")
    dgm.add_edge("X1", "X5")
    dgm.add_edge("L3", "X6")
    dgm.add_edge("L3", "X7")
    dgm.add_edge("L1", "L4")
    dgm.add_edge("L4", "X10")
    dgm.add_edge("L4", "X11")
    dgm.add_edge("L4", "X12")
    dgm.add_edge("L3", "X11")
    dgm.add_edge("L3", "X12")
    dgm.add_edge("X3", "X5")
    dgm.add_edge("X2", "X1")
    dgm.add_edge("L2", "X8")
    dgm.add_edge("L2", "X9")
    dgm.add_edge("X2", "X4")
    dgm.add_edge("X2", "X1")
    dgm.add_edge("X4", "L2")

    return dgm

def Fanning_Indicator(seed = 0): 
    dgm = LinearSCM(seed=seed)
    n_lats = 4;  n_obs = 12
    for i in range(1, n_lats + 1):
        dgm.add_variable(f"L{i}", False) # Add latent variables
    for j in range(1, n_obs + 1):
        dgm.add_variable(f"X{j}", True)  # Add observed variables
    dgm.add_edge("L1", "L2")
    dgm.add_edge("L1", "L3")
    dgm.add_edge("L1", "L4")
    for j in range(1, 6):
        dgm.add_edge("L2", f"X{j}")
    for j in range(3, 10):
        dgm.add_edge("L3", f"X{j}")
    for j in range(8, 12+1): 
        dgm.add_edge("L4", f"X{j}")
    
    return(dgm)


