import numpy as np
import argparse
from simus.data import StructuralCausalModelEg2
from src.negdro_limited import negDRO_limited
from src.methods_wrappers import *
import pickle

def run_simulation(n, gammas, adjust, mode, seed=None):
    
    SCM = StructuralCausalModelEg2()
    x_list, y_list = SCM.sample(n, mode=mode, seed=seed)
    # causal effect
    beta_star = np.array([0., 2., 0., 0.])
    
    methods_compare = ['NegDRO', 'DRIG', 'CausalDantzig']
    errors_dict = {method: [] for method in methods_compare}
    
    for gamma in gammas:
        # negDRO
        # print(f'\n Running regularization parameter {gamma}')
        b_list, w_list = negDRO_limited(x_list, y_list, gamma=gamma, lambda_reg=0.01, worst_adjust=1, worst_adjust_final=adjust, 
                    grad_threshold=1e-5, verbose=True, seed=None)
        b_neg = b_list[-1]
        error_neg = np.linalg.norm(b_neg - beta_star)
        errors_dict['NegDRO'].append(error_neg)
        
        # drig
        b_drig = drig(x_list, y_list, gamma=gamma)
        error_drig = np.linalg.norm(b_drig - beta_star)
        errors_dict['DRIG'].append(error_drig)
    
    # CausalDantzig
    b_cd = causal_dantzig_reg(x_list, y_list)
    error_cd = np.linalg.norm(b_cd - beta_star)
    errors_dict['CausalDantzig'].append(error_cd)
    
    return errors_dict

def main():
    parser = argparse.ArgumentParser(description = 'Compare gamma on different mode for limited interventions')
    parser.add_argument('--mode', type=int, required=True, help='Mode of simulations')
    parser.add_argument('--sim_round', type=int, required=True, help='Round of Simulation')
    
    args = parser.parse_args()
    mode = args.mode
    sim_round = args.sim_round
    
    num_rounds = 20
    num_simulations = 200
    simulations_per_round = num_simulations // num_rounds
    
    # Run the simulations
    print(f"Simulation with mode={mode}, and sim_round={sim_round}/{num_rounds}start running.")
    # n
    n = 10000
    # gammas
    gammas = [0, 1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 60]
    # adjust
    adjust = 0.1
    
    # run simulations
    errors_dict = {}
    for i_simulation in range((sim_round - 1) * simulations_per_round, sim_round * simulations_per_round):
        print(f"  Running simulation = {i_simulation}")
        errors_per_simu = run_simulation(n, gammas, adjust, mode, seed=i_simulation)
        errors_dict[i_simulation] = errors_per_simu
    
    results = {'errors': errors_dict}
    
    # Save the results to a file
    filename = f'results/limited_intervene_mode{mode}_simround{sim_round}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Simulation Example 2 with mode={mode}, simu_round={sim_round}completed.")

if __name__ == '__main__':
    main()