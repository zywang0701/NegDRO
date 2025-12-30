"""
Experiment 1 (gamma): Evaluation of NegDRO with different regularization parameter γ.

This script evaluates the performance of NegDRO across different values of the
regularization parameter γ in the range of [0, 20].

In the paper, the input dimension p is varied as p ∈ {5, 10, 40}, and the experiment
is run with round_num ∈ {1, 2, 3, 4, 5}. Each round processes 40 simulations (out of
200 total simulations), and for each simulation, NegDRO is run with all gamma values
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
    parser = argparse.ArgumentParser(description = 'Compare gamma with specified p.')
    parser.add_argument('--p', type=int, required=True, help='Dimension (p)')
    parser.add_argument('--roundnum', type=int, required=True, help='round num')
    
    args = parser.parse_args()
    p = args.p
    n = 20000
    sim_round = args.roundnum
    gammas = [0., 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.,2., 3.,4., 5., 7., 10., 12., 15., 17., 20.]
    
    num_rounds = 5
    num_simulations = 200
    simulations_per_round = num_simulations // num_rounds
    
    beta_star = np.zeros(p)
    beta_star[0] = 0.5
    beta_star[2] = -0.5
    
    # Run the simulations
    print(f"Simulation with p={p}, sim_round={sim_round} start running.")
    errors_dict = {gamma: [] for gamma in gammas}
    for i_simulation in range((sim_round - 1) * simulations_per_round, sim_round * simulations_per_round):
        if i_simulation % 10 == 0:
            print(f"  Running simulation = {i_simulation}")
        # generate data
        SCM = StructuralCausalModelSimu1(p)
        x_list, y_list = SCM.sample(n, mode=2, seed=i_simulation)
        for gamma in gammas:
            b_neg, _ = negDRO(x_list, y_list, gamma=gamma, early_stop=True, num_iter=1500, log_interval=100)
            b_neg_error = np.linalg.norm(b_neg - beta_star)
            errors_dict[gamma].append(b_neg_error)
    
    results = {'errors': errors_dict}
    
    # Save the results to a file
    filename = f'results/comp_gamma_p{p}_simround{sim_round}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    

if __name__ == '__main__':
    main()