import numpy as np
import torch
import cvxpy as cp

def compute_loss(x, y, b, intercept):
    """
    Compute the mean squared loss for a given environment.
    """
    if intercept:
        x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1)
    preds = x @ b
    return torch.mean((y - preds) ** 2)            

def solve_for_w(losses, lambda_reg):
    """
    Given environments' losses, solve for the optimal weight using quadratic programming
    """
    n_env = len(losses)
    w = cp.Variable(n_env)
    objective = cp.Maximize(cp.matmul(losses, w) - lambda_reg * cp.sum_squares(w))
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    w_opt = torch.tensor(w.value, dtype=torch.float32)
    val_opt = prob.value
    return w_opt, val_opt

def negDRO_solvew(x_list, y_list, gamma, intercept=False, lr_b=0.01, lambda_reg=0.001,
                  worst_adjust=1, worst_adjust_final=0.01, threshold=None, grad_threshold=1e-4, 
                  verbose=False, seed=None):
    """
    Main Algorithm for NegDRO with variable worst_group adjustment, where worst_adjust 
    decreases each step until the number of variables below the threshold stabilizes.
    Once stable, a final step with worst_adjust_final is run.

    Args:
        x_list (list): List of feature matrices (torch tensors) for each environment.
        y_list (list): List of target vectors (torch tensors) for each environment.
        gamma (float): The regularization parameter.
        intercept (bool): If True, includes an intercept term in the model. Defaults to True.
        lr_b (float): The learning rate for b.
        lambda_reg (float): The ridge penalty added on weight.
        worst_adjust (float): Initial adjusted weight on the highest environment risk to objective.
        worst_adjust_final (float): Final adjusted weight on the highest environment risk to objective.
        threshold (float): Threshold for setting coefficients to zero if their absolute value < threshold.
        grad_threshold (float): Gradient norm threshold for early stopping.
        verbose (bool): If True, prints intermediate results.
        seed (int, optional): Seed for initializing coefficients.

    Returns:
        b_list (list): List of numpy arrays containing `b` values after each step (excluding intercept if `intercept=True`).
    """
    n_env = len(x_list)
    _, p = x_list[0].shape
    n = min([x_list[e].shape[0] for e in range(n_env)])
    x_list = [torch.tensor(x, dtype=torch.float32) for x in x_list]
    y_list = [torch.tensor(y, dtype=torch.float32) for y in y_list]
    
    num_iter = 20000 # Max iteration time for each step
    if threshold is None:
        threshold = 0.25 * (p / n) ** 0.25
    mask = None  # Initialize mask to None
    
    # Initialize list to store b, w values after each step
    b_list = []
    w_list = []
    
    # Initialize list to store worst environment risk after each step
    max_loss_list = []
    
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    
    # Initialize b
    if intercept:
        limit = np.sqrt(6 / (p + p + 1))
        b = torch.empty(p + 1).uniform_(-limit, limit)
    else:
        limit = np.sqrt(6 / (p + p))
        b = torch.empty(p).uniform_(-limit, limit)
        
    step = 0
    final_step = 0
    prev_num_below_threshold = None
    
    # Function to run one optimization step with additional weight on highest environment risk
    def run_negDRO_step(b, worst_adjust_step, step, final=False):
        if verbose:
            if final:
                print(f"\nStarting Final Optimization Step {step + 1} with additional weight on worst env={worst_adjust_step:.3f}")
            else:
                print(f"\nStarting Optimization Step {step + 1} with additional weight on worst env={worst_adjust_step:.3f}")
        
        # Apply mask to freeze coefficients (set them to zeros)
        if mask is not None:
            with torch.no_grad():
                b[~mask] = 0.0
        b = torch.nn.Parameter(b)
            
        # Initialize weight vector w as a tensor without gradient tracking
        w = torch.rand(n_env)
        w = w / w.sum()
        w.requires_grad = False # # Disable gradient tracking for w
        
        # Define optimizers and schedulers
        optimizer_b = torch.optim.Adam([b], lr=lr_b)
        scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=2000, gamma=0.6)

        
        # Initialize variables to track the minimal gradient norm and corresponding b
        min_b_grad_L2 = float('inf')
        b_min_grad = b.detach().clone()
        converged = False
        
        # Optimization loop
        for i in range(num_iter):
            optimizer_b.zero_grad()
            
            # Compute losses for each environment
            losses = [compute_loss(x_list[e], y_list[e], b, intercept) for e in range(n_env)]
            losses_tensor = torch.stack(losses)
            
            # Update w by solving quadratic programming
            w, val_w = solve_for_w(losses_tensor.detach(), lambda_reg)
            w.requires_grad = False
            
            # Compute total weighted loss
            total_loss = torch.sum((w - gamma / (1 + gamma * n_env)) * losses_tensor)
            
            # Adjust the optimization weight for the worst environment
            worst_loss_adjust = worst_adjust_step * torch.mean(losses_tensor)
            total_loss = total_loss + worst_loss_adjust - lambda_reg * torch.sum(w ** 2)
            
            # Backpropagation
            total_loss.backward()
            
            # Apply gradient masking if mask is available
            if mask is not None:
                b.grad = b.grad * mask.float()
            
            # Gradient norm for early stopping
            b_grad_L2 = torch.norm(b.grad).item()
            
            # Update the minimal gradient norm and corresponding b
            if b_grad_L2 < min_b_grad_L2:
                min_b_grad_L2 = b_grad_L2
                b_min_grad = b.detach().clone()
            
            if b_grad_L2 < grad_threshold:
                converged = True
                if verbose:
                    print(f"Converged at iteration {i}, adjusted Loss: {total_loss.item():.4f}, b_grad_L2: {b_grad_L2:.5f}, Optimal val_w: {val_w:.4f}")
                    print(f"Weight vector w: {w}")
                break
            
            if b_grad_L2 > 5e3:
                converged = False
                print('Gradient explosion detected. Try smaller learning rates or a different initial point.')
                break  # Exiting the loop but will return b_min_grad
            
            # Update b using gradient descent 
            optimizer_b.step()
            scheduler_b.step()

            if i % 2000 == 0 and verbose:
                print(f"Iteration {i}, adjusted Loss: {total_loss.item():.4f}, b_grad_L2: {b_grad_L2:.5f}, Optimal val_w: {val_w:.4f}")
                # print(f"Weight vector w: {w}")
        if not converged:
            b = b_min_grad
        
        # Apply zero thresholding
        with torch.no_grad():
            if intercept:
                b_vals = b[:-1]
            else:
                b_vals = b
            below_threshold = torch.abs(b_vals) < threshold
            num_below_threshold = below_threshold.sum().item()  # Number of variables below threshold
            b_vals[below_threshold] = 0.0

        # Store the b values after this step
        b_step = b_vals.detach().numpy().flatten()

        # Update mask for the next step (excluding intercept term)
        mask_values = (np.abs(b_step) >= threshold)
        if intercept:
            mask_out = np.append(mask_values, [True])  # Keep intercept term always
        else:
            mask_out = mask_values
        new_mask = torch.tensor(mask_out, dtype=torch.bool, device=b.device)
        
        # Store the maximum loss
        b_test = torch.from_numpy(b_step).float()
        b_test[~new_mask] = 0.0
        losses = [compute_loss(x_list[e], y_list[e], b_test, intercept) for e in range(n_env)]
        losses_tensor = torch.stack(losses)
        max_loss = torch.max(losses_tensor).item()

        if verbose:
            print(f"Step {step + 1}: Number of variables below threshold: {num_below_threshold}, Highest Env risk: {max_loss:.5f}.")

        return b, b_step, w, num_below_threshold, max_loss, new_mask
    
    # Primary Loop: Iterative Adjustment
    while True:
        worst_adjust_step = worst_adjust / (2**step)
        b, b_step, w, num_below_threshold, max_loss, mask = run_negDRO_step(b, worst_adjust_step, step)
        b_list.append(b_step)
        w_list.append(w)
        max_loss_list.append(max_loss)
        
        if prev_num_below_threshold is not None:
            if num_below_threshold == prev_num_below_threshold:
                # No change in below-threshold variables: break and run final step with worst_adjust_final
                if verbose:
                    print(f"No change in below-threshold variables at step {step + 1}. Running final steps with worst_adjust {worst_adjust_final}")
                break
        
        prev_num_below_threshold = num_below_threshold
        step += 1
    
    # Secondary Loop: Final adjustment with worst_adjust_final
    while True:
        
        # run the final step with the worst_adjust_final
        b, b_step, w, num_below_threshold_new, max_loss, mask = run_negDRO_step(b, worst_adjust_final, final_step, final=True)
        b_list.append(b_step)
        w_list.append(w)
        max_loss_list.append(max_loss)

        # Check if num_below_threshold has changed
        if num_below_threshold_new == num_below_threshold:
            if verbose:
                print(f"No change in below-threshold variables after step {final_step+1}. Stopping adjustments.")
            break  # Exit the secondary loop as no change occurred

        # Update for the next iteration
        num_below_threshold = num_below_threshold_new
        final_step += 1

    return b_list, w_list


def negDRO_solvew_singlestep(x_list, y_list, gamma, intercept=False, lr_b=0.01, lambda_reg=0.001,
                             worst_adjust=1, grad_threshold=1e-4, verbose=False, seed=None):
    """
    Main Algorithm for negative DRoL with variable L2 penalty, where L2 penalty 
    decreases each step until the number of variables below the threshold stabilizes.
    Once stable, a final step with l2_penalty=0 is run.

    Args:
        x_list (list): List of feature matrices (torch tensors) for each environment.
        y_list (list): List of target vectors (torch tensors) for each environment.
        gamma (float): The regularization parameter.
        intercept (bool): If True, includes an intercept term in the model. Defaults to True.
        lr_b (float): The learning rate for b.
        lambda_reg (float): The ridge penalty added on weight.
        worst_adjust (float): Initial adjusted weight on the highest environment risk to objective.
        threshold (float): Threshold for setting coefficients to zero if their absolute value < threshold.
        verbose (bool): If True, prints intermediate results.
        seed (int, optional): Seed for initializing coefficients.

    Returns:
        b_list (list): List of numpy arrays containing `b` values after each step (excluding intercept if `intercept=True`).
    """
    n_env = len(x_list)
    _, p = x_list[0].shape
    n = min([x_list[e].shape[0] for e in range(n_env)])
    x_list = [torch.tensor(x, dtype=torch.float32) for x in x_list]
    y_list = [torch.tensor(y, dtype=torch.float32) for y in y_list]
    
    num_iter = 20000 # Max iteration time for each step
    
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    
    # Initialize b
    if intercept:
        limit = np.sqrt(6 / (p + p + 1))
        b = torch.empty(p + 1).uniform_(-limit, limit)
    else:
        limit = np.sqrt(6 / (p + p))
        b = torch.empty(p).uniform_(-limit, limit)
    b = torch.nn.Parameter(b)
    
    # Initialize weight vector w as a tensor without gradient tracking
    w = torch.rand(n_env)
    w = w / w.sum()
    w.requires_grad = False # # Disable gradient tracking for w
    
    # Define optimizers and schedulers
    optimizer_b = torch.optim.Adam([b], lr=lr_b)
    scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=2000, gamma=0.6)
    
    # Initialize variables to track the minimal gradient norm and corresponding b
    min_b_grad_L2 = float('inf')
    b_min_grad = b.detach().clone()
    converged = False
    
    # Optimization loop
    for i in range(num_iter):
        optimizer_b.zero_grad()
        
        # Compute losses for each environment
        losses = [compute_loss(x_list[e], y_list[e], b, intercept) for e in range(n_env)]
        losses_tensor = torch.stack(losses)
        
        # Update w by solving quadratic programming
        w, val_w = solve_for_w(losses_tensor.detach(), lambda_reg)
        w.requires_grad = False
        
        # Compute total weighted loss
        total_loss = torch.sum((w - gamma / (1 + gamma * n_env)) * losses_tensor)
        
        # Adjust the optimization weight for the worst environment
        worst_loss_adjust = worst_adjust * torch.max(losses_tensor)
        total_loss = total_loss + worst_loss_adjust - lambda_reg * torch.sum(w ** 2)
        
        # Backpropagation
        total_loss.backward()
        

        # Gradient norm for early stopping
        b_grad_L2 = torch.norm(b.grad).item()
        
        # Update the minimal gradient norm and corresponding b
        if b_grad_L2 < min_b_grad_L2:
            min_b_grad_L2 = b_grad_L2
            b_min_grad = b.detach().clone()
        
        if b_grad_L2 < grad_threshold:
            converged = True
            if verbose:
                print(f"Converged at iteration {i}, adjusted Loss: {total_loss.item():.4f}, b_grad_L2: {b_grad_L2:.5f}, Optimal val_w: {val_w:.4f}")
                print(f"Weight vector w: {w}")
            break
        
        if b_grad_L2 > 5e3:
            converged = False
            print('Gradient explosion detected. Try smaller learning rates or a different initial point.')
            break  # Exiting the loop but will return b_min_grad
        
        # Update b using gradient descent 
        optimizer_b.step()
        scheduler_b.step()

        if i % 2000 == 0 and verbose:
            print(f"Iteration {i}, adjusted Loss: {total_loss.item():.4f}, b_grad_L2: {b_grad_L2:.5f}, Optimal val_w: {val_w:.4f}")
            # print(f"Weight vector w: {w}")
    if not converged:
        b = b_min_grad
    
    return b.detach().numpy().flatten()