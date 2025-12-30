"""
Experiment 1 (T): Evaluation of NegDRO convergence trajectory over iterations T.

This script evaluates the convergence behavior of NegDRO by tracking the coefficient
trajectory over iterations. The experiment uses fixed parameters: sample size n = 20000
and regularization parameter γ = 20.0, with early stopping disabled to capture the
full optimization trajectory.

For each run, NegDRO performs 1500 iterations with coefficient logging every 50
iterations (log_interval=50), recording the coefficient vector at each checkpoint.
The error (L2 norm) between the coefficient trajectory and the true beta_star is
computed to analyze convergence behavior.

In the paper, the input dimension p is varied as p ∈ {5, 10, 40}, and the experiment
is run with round_num ∈ {1, 2, 3, 4, 5}. Each round processes 40 simulations (out of
200 total simulations), all using the same data generation seed but different random
seeds for NegDRO optimization.

The results are saved separately for each p and round_num combination.
"""

import numpy as np
import argparse
from simus.data import StructuralCausalModelSimu1
from src.negdro import negDRO
import pickle

def main():
    parser = argparse.ArgumentParser(description = 'Compare T with specified p.')
    parser.add_argument('--p', type=int, required=True, help='Dimension (p)')
    parser.add_argument('--roundnum', type=int, required=True, help='round num')
    
    args = parser.parse_args()
    p = args.p
    n = 20000
    sim_round = args.roundnum
    gamma = 20.
    
    num_rounds = 5
    num_simulations = 200
    simulations_per_round = num_simulations // num_rounds
    
    beta_star = np.zeros(p)
    beta_star[0] = 0.5
    beta_star[2] = -0.5
    
    # Run the simulations
    SCM = StructuralCausalModelSimu1(p)
    x_list, y_list = SCM.sample(n, mode=2, seed=1)
    print(f"Simulation with p={p}, sim_round={sim_round} start running.")
    b_traj_list = []
    for i_simulation in range((sim_round - 1) * simulations_per_round, sim_round * simulations_per_round):
        if i_simulation % 10 == 0:
            print(f"  Running simulation = {i_simulation}")
        _, b_traj = negDRO(x_list, y_list, gamma=gamma, early_stop=False, num_iter=1500, log_interval=50)
        b_traj_list.append(b_traj)
        
    b_traj_array = np.array(b_traj_list)
    b_traj_norm = np.linalg.norm(b_traj_array - beta_star, axis=2)
    
    results = {'errors': b_traj_norm}
    
    # Save the results to a file
    filename = f'results/comp_T_p{p}_simround{sim_round}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    

if __name__ == '__main__':
    main()