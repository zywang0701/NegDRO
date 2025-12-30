"""
Experiment 1 (n): Evaluation of NegDRO with different sample sizes n.

This script evaluates the performance of NegDRO across different sample sizes
n ∈ {500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000},
with a fixed regularization parameter γ = 20.0.

In the paper, the input dimension p is varied as p ∈ {5, 10, 40}, and the experiment
is run with round_num ∈ {1, 2, 3, 4, 5}. Each round processes 40 simulations (out of
200 total simulations), and for each simulation, NegDRO is run with all sample sizes
to compute the error (L2 norm) between the estimated coefficient vector and the true
beta_star = [0.5, 0, -0.5, 0, ...].

The results are saved separately for each p and round_num combination.
"""

import numpy as np
import argparse
from simus.data import StructuralCausalModelSimu1
from src.negdro import negDRO
import pickle

def main():
    parser = argparse.ArgumentParser(description = 'Compare n with specified p.')
    parser.add_argument('--p', type=int, required=True, help='Dimension (p)')
    parser.add_argument('--roundnum', type=int, required=True, help='round num')
    
    args = parser.parse_args()
    p = args.p
    ns = [500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    sim_round = args.roundnum
    gamma = 20.
    
    num_rounds = 5
    num_simulations = 200
    simulations_per_round = num_simulations // num_rounds
    
    beta_star = np.zeros(p)
    beta_star[0] = 0.5
    beta_star[2] = -0.5
    
    # Run the simulations
    print(f"Simulation with p={p}, sim_round={sim_round} start running.")
    errors_dict = {n: [] for n in ns}
    for i_simulation in range((sim_round - 1) * simulations_per_round, sim_round * simulations_per_round):
        if i_simulation % 10 == 0:
            print(f"  Running simulation = {i_simulation}")
        # generate data
        for n in ns:
            SCM = StructuralCausalModelSimu1(p)
            x_list, y_list = SCM.sample(n, mode=2, seed=i_simulation)
            b_neg, _ = negDRO(x_list, y_list, gamma=gamma, early_stop=True, num_iter=1500, log_interval=100)
            b_neg_error = np.linalg.norm(b_neg - beta_star)
            errors_dict[n].append(b_neg_error)
    
    results = {'errors': errors_dict}
    
    # Save the results to a file
    filename = f'results/comp_n_p{p}_simround{sim_round}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    

if __name__ == '__main__':
    main()