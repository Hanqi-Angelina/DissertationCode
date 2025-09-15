from itertools import permutations
import numpy as np
import pandas as pd
from convert_to_CPDAG import signed_to_CPDAG

def reorder_adj_matrix_to_order(adj_matrix, var_names, observed_order_prefix='X', observed_target_order=None):
    """
    Reorder so observed variables (names starting with 'X') are first and (if provided)
    follow observed_target_order. Latents follow in their original relative order.
    Returns reordered_matrix, reordered_names.
    """
    # split names
    observed_vars = [name for name in var_names if name.startswith(observed_order_prefix)]
    latent_vars   = [name for name in var_names if not name.startswith(observed_order_prefix)]

    if observed_target_order is None:
        sorted_var_names = observed_vars + latent_vars
    else:
        # put observed in the target order
        in_target = [nm for nm in observed_target_order if nm in observed_vars]
        sorted_var_names = in_target + latent_vars

    permutation = [var_names.index(name) for name in sorted_var_names]
    reordered_matrix = adj_matrix[permutation][:, permutation]
    return reordered_matrix, sorted_var_names

def calculate_shd(learned_H, true_G):
    """
    Calculates the Structural Hamming Distance (SHD) between two graphs
    using a concise comparison.

    Args:
        learned_H (np.ndarray): The matrix for the learned graph.
        true_G (np.ndarray): The matrix for the true graph.

    Returns:
        int: The calculated SHD value.
    """
    if learned_H.shape != true_G.shape:
        raise ValueError("Matrices must have the same dimensions.")
    
    shd = 0
    n = learned_H.shape[0]

    for i in range(n):
        for j in range(i + 1, n):  # Check each unique pair of nodes once
            
            # Get the matrix entries for the edge between i and j in both graphs
            learned_pair = (learned_H[i, j], learned_H[j, i])
            true_pair = (true_G[i, j], true_G[j, i])
            
            # If the edge type/direction in the learned graph does not match
            # the edge type/direction in the true graph, it's a single mismatch.
            if learned_pair != true_pair:
                shd += 1
                
    return shd

def convert_to_undirected(m):
    abs_matrix = np.abs(m)
    # Symmetrize the matrix
    symmetric_matrix = np.logical_or(abs_matrix, abs_matrix.T)
    converted_matrix = symmetric_matrix.astype(int)
    
    return converted_matrix

def calculate_f1(m1, m2):
    # calculate f1 for the skeleton of discovered structure
    tp = np.sum((m1 !=0 ) & (m2 != 0))
    fp = np.sum((m1 == 0) & (m2 != 0))
    fn = np.sum((m1 != 0) & (m2 == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def pad_bottom_right(A, target, all_vars):
    n = A.shape[0]
    if n < target: 
        out = np.zeros((target, target), dtype=A.dtype)
        out[:n, :n] = A
        new_var = all_vars.copy()
        new_var.extend([f"A{i}" for i in range(1, target-n+1)])
        return out, new_var
    else: 
        return A, all_vars

def get_shd_f1(ground_truth_matrix, predicted_matrix, n_x, method):
    '''
    Input: ground truth matrix (DataFrame), predicted matrix (DataFrame/ndarray/tensor), number of observed variables
    Output: SHD, F1 score, permutation of the predicted latent block that best matches the ground truth
    '''

    # --- Preprocessing and Standardization ---

    # Handle ground_truth_matrix (DataFrame)

    var_names1 = list(ground_truth_matrix.columns)
    ground_truth_matrix = ground_truth_matrix.to_numpy()

    # Handle predicted_matrix (predicted)
    print(predicted_matrix)
    var_names2 = list(predicted_matrix.columns)
    predicted_matrix = predicted_matrix.to_numpy()

    # --- Name alignment: put the SAME observed names at the front in the SAME order ---
    obs1 = [nm for nm in var_names1 if nm.startswith('X')]
    observed_target_order = obs1.copy()

    ground_truth_matrix, var_names1 = reorder_adj_matrix_to_order(ground_truth_matrix, var_names1,
                                                      observed_order_prefix='X',
                                                      observed_target_order=observed_target_order)
    predicted_matrix, var_names2 = reorder_adj_matrix_to_order(predicted_matrix, var_names2,
                                                      observed_order_prefix='X',
                                                      observed_target_order=observed_target_order)

    # --- Pad both matrices to the same size ---
    n = max(ground_truth_matrix.shape[0], predicted_matrix.shape[0])
    ground_truth_matrix, var_names1 = pad_bottom_right(ground_truth_matrix, n, var_names1)
    # print(ground_truth_matrix); print(var_names1)
    predicted_matrix, var_names2 = pad_bottom_right(predicted_matrix, n, var_names2)
    # print(predicted_matrix); print(var_names2)

    # --- Permutation search over the LATENT block only (prediction side) ---
    n_z = max(0, n - n_x)
    # print(n_z)
    if n_z > n:
        raise ValueError("n_z is larger than the matrix size after preprocessing")

    perms = list(permutations(range(n_z)))
    # print(perms)
    min_shd = float('inf')
    max_f1 = -float('inf')
    best_perm = None
    
    ground_truth_undirected = convert_to_undirected(ground_truth_matrix)
    # print(ground_truth_undirected)
    predicted_matrix_undirected = convert_to_undirected(predicted_matrix)
    # print(predicted_matrix_undirected)
    ground_truth_CPDAG = signed_to_CPDAG(ground_truth_matrix, var_names1)
    if method == "RLCD" or method == "LaHME": 
        predicted_CPDAG = signed_to_CPDAG(predicted_matrix, var_names2)
    else: 
        predicted_CPDAG = predicted_matrix
    for perm in perms:
        # observed fixed [0..n_x-1], permute latent tail [n_x..n-1]
        full_perm_pred = list(range(n_x)) + [n_x + p for p in perm]
        # print(full_perm_pred)
        permuted_CPDAG = predicted_CPDAG[full_perm_pred][:, full_perm_pred]
        permuted_undirected = predicted_matrix_undirected[full_perm_pred][:, full_perm_pred]
        current_shd = calculate_shd(ground_truth_CPDAG, permuted_CPDAG)
        # print(current_shd)
        current_f1 = calculate_f1(ground_truth_undirected, permuted_undirected) # calculate based on skeleton of graph

        if (current_shd == min_shd and current_f1 > max_f1) or current_shd < min_shd:
            min_shd = current_shd
            max_f1 = current_f1
            best_perm = list(perm)
    best_perm_pred = list(range(n_x)) + [n_x + p for p in best_perm]
    permuted_CPDAG = predicted_CPDAG[best_perm_pred][:, best_perm_pred]
    permuted_vars = [var_names2[i] for i in best_perm_pred]
    best_perm_df = pd.DataFrame(permuted_CPDAG, index= permuted_vars, columns= permuted_vars)
    return min_shd, max_f1, best_perm_df

