import numpy as np
from scipy.optimize import linprog

def cDantzig(lambda_, X, Y, ExpInd, weights=None):
    """
    Regularized causal Dantzig selector.

    Parameters:
    - lambda_: Regularization parameter (scalar).
    - X: Predictor variables (numpy array of shape (n_samples, n_features)).
    - Y: Response variable (numpy array of shape (n_samples,)).
    - ExpInd: Experimental indices (numpy array of shape (n_samples,)).
    - weights: Optional weights (default is None).

    Returns:
    - coefficients: Estimated coefficients (numpy array of shape (n_features,)).
    """
    ExpInd = ExpInd.astype(str)
    unique_ExpInd = np.unique(ExpInd)
    n_groups = len(unique_ExpInd)
    p = X.shape[1]

    # Compute group-specific covariance matrices
    foo = {}
    for e in unique_ExpInd:
        X_e = X[ExpInd == e]
        n_e = X_e.shape[0]
        foo[e] = X_e.T @ X_e / n_e

    total_foo = sum(foo.values())

    # Compute differences in covariance matrices
    cov_XX_diff = []
    for e in unique_ExpInd:
        other_foo_sum = total_foo - foo[e]
        cov_XX_diff_e = foo[e] - other_foo_sum / (n_groups - 1)
        cov_XX_diff.append(cov_XX_diff_e)

    diffX = np.vstack(cov_XX_diff)

    # Compute X multiplied by Y
    XxY = X * Y[:, np.newaxis]

    # Compute group-specific means of XxY
    foo_Y = {}
    for e in unique_ExpInd:
        XxY_e = XxY[ExpInd == e]
        foo_Y[e] = XxY_e.mean(axis=0)

    total_foo_Y = sum(foo_Y.values())

    # Compute differences in covariance between X and Y
    cov_XY_diff = []
    for e in unique_ExpInd:
        other_foo_Y_sum = total_foo_Y - foo_Y[e]
        cov_XY_diff_e = foo_Y[e] - other_foo_Y_sum / (n_groups - 1)
        cov_XY_diff.append(cov_XY_diff_e)

    diffY = np.concatenate(cov_XY_diff)

    # Construct the constraint matrices for the linear program
    Atop = np.hstack([-diffX, diffX])
    A = np.vstack([Atop, -Atop])
    btop = -diffY

    if weights is None:
        b = np.concatenate([btop + lambda_, -btop + lambda_])
    else:
        expanded_weights = np.concatenate([weights[e] for e in unique_ExpInd])
        weights_clipped = np.maximum(expanded_weights, 0.1)
        b = np.concatenate([btop + lambda_ * weights_clipped,
                            -btop + lambda_ * weights_clipped])

    c = np.ones(2 * p)

    # Solve the linear programming problem
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

    if not res.success:
        raise ValueError(f"Linear programming failed: {res.message}")

    x = res.x
    coefficients = x[:p] - x[p:]

    return coefficients


def compute_loss_cDantzig(lambda_, X, Y, ExpInd, weights=None):
    """
    Estimates the loss of cDantzig for a given lambda via cross-validation.

    Parameters:
    - lambda_: Regularization parameter (scalar).
    - X: Predictor variables (numpy array of shape (n_samples, n_features)).
    - Y: Response variable (numpy array of shape (n_samples,)).
    - ExpInd: Experimental indices (numpy array of shape (n_samples,)).
    - weights: Optional weights (dictionary or None).

    Returns:
    - mean_residual: Mean residual over folds (scalar).
    """
    ExpInd = ExpInd.astype(str)
    unique_ExpInd = np.unique(ExpInd)
    n_samples, n_features = X.shape
    folds = 10
    residuals = np.zeros(folds)

    for i in range(folds):
        select = np.ones(n_samples, dtype=bool)

        # Stratified cross-validation: ensure each group is represented in each fold
        for e in unique_ExpInd:
            indices_e = np.where(ExpInd == e)[0]
            samples = len(indices_e)
            fold_size = samples // folds
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < folds - 1 else samples
            selection_indices = indices_e[start_idx:end_idx]
            select[selection_indices] = False  # Validation indices

        # Training data
        X_train, Y_train, ExpInd_train = X[select], Y[select], ExpInd[select]
        # Validation data
        X_val, Y_val, ExpInd_val = X[~select], Y[~select], ExpInd[~select]

        # Prepare weights if not provided
        if weights is None:
            weights_dict = {e: np.ones(n_features) for e in np.unique(ExpInd_val)}
        else:
            weights_dict = weights

        # Try to compute gamma using cDantzig
        try:
            gamma = cDantzig(lambda_, X_train, Y_train, ExpInd_train, weights=weights)
        except ValueError:
            gamma = np.array([])

        if gamma.size != 0:
            # Compute residuals on validation data
            residual = compute_validation_residual(X_val, Y_val, ExpInd_val, gamma, weights_dict)
            residuals[i] = residual
        else:
            residuals[i] = np.inf  # Assign infinity if gamma could not be computed

    mean_residual = np.mean(residuals)
    return mean_residual


def compute_validation_residual(X_val, Y_val, ExpInd_val, gamma, weights_dict):
    """
    Computes the residual for the validation data.

    Parameters:
    - X_val: Validation predictors (numpy array).
    - Y_val: Validation response (numpy array).
    - ExpInd_val: Validation experimental indices (numpy array).
    - gamma: Coefficients from cDantzig (numpy array).
    - weights_dict: Weights dictionary.

    Returns:
    - maxi: Maximum weighted absolute difference (scalar).
    """
    residuals = Y_val - X_val @ gamma
    XxY = X_val * residuals[:, np.newaxis]

    # Group-wise computations
    unique_Ep = np.unique(ExpInd_val)
    foo = {}
    for e in unique_Ep:
        XxY_e = XxY[ExpInd_val == e]
        foo[e] = XxY_e.mean(axis=0)

    cov_XY_diff_abs_weighted = []
    for e in unique_Ep:
        other_foos = [foo[other_e] for other_e in unique_Ep if other_e != e]
        if len(other_foos) > 0:
            mean_other_foos = np.mean(other_foos, axis=0)
            diff = np.abs(foo[e] - mean_other_foos) / weights_dict[e]
            cov_XY_diff_abs_weighted.append(diff)
        else:
            # If there's only one group, avoid division by zero
            diff = np.abs(foo[e]) / weights_dict[e]
            cov_XY_diff_abs_weighted.append(diff)

    # Maximum of the weighted absolute differences
    maxi = np.max(np.concatenate(cov_XY_diff_abs_weighted))
    return maxi


def causalDantzig(X, Y, ExpInd):
    """
    Estimates causal effects using the Dantzig selector.

    Parameters:
    - X: Predictor variables (numpy array of shape (n_samples, n_features)).
    - Y: Response variable (numpy array of shape (n_samples,)).
    - ExpInd: Experimental indices (numpy array of shape (n_samples,)).

    Returns:
    - coefficients: Estimated coefficients (numpy array of shape (n_features,)).
    """
    # Ensure X, Y, and ExpInd are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    ExpInd = np.asarray(ExpInd).astype(str)

    unique_ExpInd = np.unique(ExpInd)
    n_groups = len(unique_ExpInd)
    n_samples, n_features = X.shape

    # Normalize X and Y to overall mean zero
    XY = np.hstack((X, Y.reshape(-1, 1)))
    group_means = []
    for e in unique_ExpInd:
        XY_e = XY[ExpInd == e]
        mean_e = XY_e.mean(axis=0)
        group_means.append(mean_e)
    group_means = np.vstack(group_means)
    overall_mean = group_means.mean(axis=0)
    X = X - overall_mean[:n_features]
    Y = Y - overall_mean[n_features]

    # Compute per-group means of X^2
    foo = {}
    for e in unique_ExpInd:
        X_e = X[ExpInd == e]
        foo[e] = (X_e ** 2).mean(axis=0)

    # Compute weights for each group
    weights = {}
    for e in unique_ExpInd:
        foo_e = foo[e]
        other_foos = [foo[key] for key in foo if key != e]
        if len(other_foos) > 0:
            mean_other_foos = np.mean(other_foos, axis=0)
        else:
            mean_other_foos = np.zeros_like(foo_e)
        weight_e = np.sqrt(foo_e + (mean_other_foos) ** 2)
        weights[e] = weight_e

    # Define lambda sequence
    XxY = X * Y[:, np.newaxis]
    foo_cov = {}
    for e in unique_ExpInd:
        XxY_e = XxY[ExpInd == e]
        foo_cov[e] = XxY_e.mean(axis=0)

    cov_XY_diff_abs_weighted = []
    for e in unique_ExpInd:
        foo_e = foo_cov[e]
        other_foos = [foo_cov[key] for key in foo_cov if key != e]
        if len(other_foos) > 0:
            mean_other_foos = np.mean(other_foos, axis=0)
        else:
            mean_other_foos = np.zeros_like(foo_e)
        diff = np.abs(foo_e - mean_other_foos) / weights[e]
        cov_XY_diff_abs_weighted.append(diff)

    # Compute maxi and mini for lambda sequence
    cov_XY_diff_abs_weighted_array = np.concatenate(cov_XY_diff_abs_weighted)
    maxi = np.max(cov_XY_diff_abs_weighted_array)
    mini = maxi * 0.001
    lambdasequence = mini * np.exp(np.arange(30) * np.log(maxi / mini) / 29)

    # Compute cross-validation loss for each lambda
    crossv = np.array([compute_loss_cDantzig(lambda_i, X, Y, ExpInd, weights=weights) for lambda_i in lambdasequence])
    lambda_selected = lambdasequence[np.argmin(crossv)]

    # Compute coefficients using the selected lambda
    coefficients = cDantzig(lambda_selected, X, Y, ExpInd, weights=weights)

    return coefficients