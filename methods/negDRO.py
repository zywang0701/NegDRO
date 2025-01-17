import numpy as np
from negDRO_solvew import negDRO_solvew, negDRO_solvew_singlestep

def negDRO(x_list, y_list, gamma, intercept=False, lr_b=0.01, lambda_reg=0.001,
           worst_adjust=1, grad_threshold=1e-4, run_threshold=0.2, runs=10, seed=None, verbose=False):
    """
    Main Algorithm for negDRO. It runs negDRO_solvew_singlestep multiple times with different initial seeds.
    If these runs do converge to the same solution, it returns the solution, meaning that the multi-env settings
        satisfy the Condition 2 in the paper.
    Otherwise, it runs negDRO_solvew to get the solution.

    Args:
        x_list (list): List of feature matrices (torch tensors) for each environment.
        y_list (list): List of target vectors (torch tensors) for each environment.
        gamma (float): The regularization parameter.
        intercept (bool): If True, includes an intercept term in the model. Defaults to True.
        lr_b (float): The learning rate for b.
        lambda_reg (float): The ridge penalty added on weight.
        worst_adjust (float): Initial adjusted weight on the highest environment risk to objective.
        threshold (float): Threshold for setting coefficients to zero if their absolute value < threshold.
        runs (int): Number of runs with different seeds to check if the solution is stable.
        seed (int, optional): Seed for initializing runs.
        verbose (bool): If True, prints intermediate results.

    Returns: 
        Depending on the stability of the solutions from multiple runs, it returns either the mean solution or the list of solutions.
        - b_mean (numpy array): The mean of the solutions from multiple runs.
        - b_list (list): List of numpy arrays containing `b` values after each step (excluding intercept if `intercept=True`).
    """
    # Generate multiple seeds for different runs
    if seed is not None:
        np.random.seed(seed)
    seeds_run = np.random.randint(0, 1000, runs)
    
    # Run multiple runs
    b_list_runs = []
    for i in range(runs):
        b_run = negDRO_solvew_singlestep(x_list, y_list, gamma, intercept, lr_b, lambda_reg, 
                                         worst_adjust=0.0, grad_threshold=grad_threshold, verbose=verbose, seed=seeds_run[i])
        b_list_runs.append(b_run)
        
    # Check if the solutions are stable
    b_array = np.stack(b_list_runs, axis=0)
    b_mean = np.mean(b_array, axis=0)
    distances = [np.linalg.norm(b - b_mean) for b in b_list_runs]
    avg_dist = np.mean(distances)
    if avg_dist > run_threshold / gamma:
        stable = False
    else:
        stable = True
    
    # If the solutions are stable, return the mean solution
    if stable:
        if verbose:
            print("The solutions are stable. The Conditions are considered satisfied.")
        return b_mean
    else:
        if verbose:
            print("The solutions are not stable. Running the full algorithm.")
        b_list = negDRO_solvew(x_list, y_list, gamma, intercept, lr_b, lambda_reg, worst_adjust=worst_adjust, 
                               grad_threshold=grad_threshold, verbose=verbose)
        return b_list