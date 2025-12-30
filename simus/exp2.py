"""
Experiment 2: Comparison of NegDRO with alternative methods (ERM, ICP, EILLS, Anchor)
by varying the dimension p.

This script compares NegDRO against alternative methods across different
dimensions. For each dimension p ∈ {5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100},
the script:
- Runs each method on synthetic data
- Monitors computation time with a timeout limit of 30 minutes (1800 seconds)
- If a method exceeds the time limit, it terminates and skips that method for all larger p values
- Records the obtained coefficient vector b for each method
- Saves results including b_results, time_results, and method timeout information

The experiment supports two modes:
- Mode 1: with hidden confounder
- Mode 2: no hidden confounder (default)
"""

import numpy as np
from simus.data import StructuralCausalModelSimu1
from src.methods_wrappers import *
from src.negdro import negDRO
import time
import pickle
import argparse
from multiprocessing import Process, Queue
import traceback

def wrapper(q, func, args, kwargs):
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time
        q.put((True, result, time_taken))
    except Exception as e:
        end_time = time.time()
        time_taken = end_time - start_time
        q.put((False, traceback.format_exc(), time_taken))

# Define the function to run methods with a timeout
def run_with_timeout(func, args=(), kwargs={}, timeout=1800):
    """
    Run func(*args, **kwargs) with a timeout in seconds.
    Returns: (success, result, time_taken)
        - success=True: completed normally
        - success=False and result='TimeLimitExceeded': timeout
        - success=False and result=<traceback string>: error
    """
    q = Queue()
    p = Process(target=wrapper, args=(q, func, args, kwargs))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return (False, 'TimeLimitExceeded', None)
    if not q.empty():
        success, result, time_taken = q.get()
        return (success, result, time_taken)
    return (False, 'No result returned', None)
        
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run simulations for varying dimension.')
    parser.add_argument('--mode', type=int, default=2, help='Mode of simulation.')
    parser.add_argument('--round_num', type=int, default=1, help='Round number (starting from 1).')
    parser.add_argument('--num_repeats_per', type=int, default=20, help='Number of repeats per p value (default: 20).')
    args = parser.parse_args()
    
    # extract arguments
    mode = args.mode # # mode 1: with hidden confounder; mode 2: no hidden confounder
    round_num = args.round_num # from 1 to 10, indicating the current repeat index
    num_repeats_per = args.num_repeats_per # default is 20

    p_list = [5,10,15,20,25,30,35,40,50,60,70,80,90,100]
    time_limit = 1800  # 30 minutes

    methods_compare = ['NegDRO', 'ERM', 'ICP', 'EILLS', 'Anchor']
    # Track the first p where each method times out
    method_timeout_p = {m: None for m in methods_compare}
    
    # Initialize results dictionary
    b_results = {method: {p: [] for p in p_list} for method in methods_compare}
    time_results = {method: {p: [] for p in p_list} for method in methods_compare}
    

    for p in p_list:
        for repeat in range(num_repeats_per):
            seed = (round_num-1) * num_repeats_per + repeat
            print(f"Processing p={p}, repeat={repeat}...")
            
            SCM = StructuralCausalModelSimu1(p=p)
            x_list, y_list = SCM.sample(n=20000, mode=mode, seed=seed)
            beta_star = np.zeros(p)
            beta_star[0], beta_star[2] = 0.5, -0.5

            method_runners = {
                'NegDRO': lambda: negDRO(x_list, y_list, gamma=20, early_stop=True, seed=seed, num_iter=1500)[0],
                'ERM':    lambda: erm(x_list, y_list),
                'ICP':    lambda: oracle_icp(x_list, y_list, beta_star),
                'EILLS':  lambda: eills(x_list, y_list),
                'Anchor': lambda: oracle_anchor(x_list, y_list, beta_star),
            }

            for method in methods_compare:
                # skip method if it has previously timed out at smaller p
                if method_timeout_p[method] is not None and p >= method_timeout_p[method]:
                    print(f"  [{method}] skipped (previously timed out at p={method_timeout_p[method]})")
                    b_results[method][p].append(None)
                    time_results[method][p].append(None)
                    print(f"  Skipping {method} for p={p} due to previous timeout.")
                    continue
                
                func = method_runners[method]
                success, result, time_taken = run_with_timeout(func, timeout=time_limit)
                
                if success:
                    b = np.array(result)
                    b_results[method][p].append(b)
                    time_results[method][p].append(time_taken)
                else:
                    # handle timeout or error
                    if result == 'TimeLimitExceeded':
                        method_timeout_p[method] = p
                        print(f"  [{method}] timed out at p={p} -> all larger p will be skipped.")
                        b_results[method][p].append(None)
                        time_results[method][p].append(None)
                    else:
                        print(f"  [{method}] error at p={p}: {result}")
                        b_results[method][p].append(None)
                        time_results[method][p].append(None)
                        
            save_path = f'results/main_varyp_mode{mode}_round{round_num}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'b_results': b_results, 
                    'time_results': time_results,
                    'method_timeout_p': method_timeout_p
                    }, f)
            print(f"  Results saved to {save_path}.")

if __name__ == '__main__':
    main()

