# This part is adapted from IRM

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

from torch.autograd import grad


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


class InvariantRiskMinimization(object):
    def __init__(self, environments, args):
        self.train(environments, args, reg=args["reg"])

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w,
                                create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                      reg,
                                                                      error,
                                                                      penalty,
                                                                      w_str))

    def solution(self):
        return (self.phi @ self.w).view(-1, 1)


class InvariantCausalPrediction(object):
    def __init__(self, environments, args):
        self.coefficients = None
        self.alpha = args["alpha"]

        x_all = []
        y_all = []
        e_all = []

        for e, (x, y) in enumerate(environments):
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            e_all.append(np.full(x.shape[0], e))

        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        e_all = np.hstack(e_all)

        dim = x_all.shape[1]

        accepted_subsets = []
        for subset in self.powerset(range(dim)):
            if len(subset) == 0:
                continue

            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()

                p_values.append(self.mean_var_test(res_in, res_out))

            # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            p_value = min(p_values) * len(environments) - 1

            if p_value > self.alpha:
                accepted_subsets.append(set(subset))
                if args["verbose"]:
                    print("Accepted subset:", subset)

        if len(accepted_subsets):
            #print(accepted_subsets)
            accepted_features = list(set.intersection(*accepted_subsets))
            if args["verbose"]:
                print("Intersection:", accepted_features)
            self.coefficients = np.zeros(dim)

            if len(accepted_features):
                x_s = x_all[:, list(accepted_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                self.coefficients[list(accepted_features)] = reg.coef_

            self.coefficients = torch.Tensor(self.coefficients)
        else:
            self.coefficients = torch.zeros(dim)

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                    x.shape[0] - 1,
                                    y.shape[0] - 1)

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def solution(self):
        return self.coefficients.view(-1, 1)


# class EmpiricalRiskMinimizer(object):
#     def __init__(self, environments, args):
#         x_all = torch.cat([x for (x, y) in environments]).numpy()
#         y_all = torch.cat([y for (x, y) in environments]).numpy()

#         w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
#         self.w = torch.Tensor(w).view(-1, 1)

#     def solution(self):
#         return self.w


from sklearn.linear_model._base import LinearModel
class AnchorRegression(LinearModel):
    def __init__(self, lamb=1, fit_intercept=False, normalize=False, copy_X=False):
        self.lamb = lamb
        self.fit_intercept=fit_intercept
        self.normalize=normalize
        self.copy_X = copy_X

    def fit(self, X, y, A=None):
        # X, y = self._validate_data(X, y, y_numeric=True)

        if type(A) is not np.ndarray:
            A = A.values

        # Center A
        A = A - A.mean(axis=0)

        self.coef_ = np.linalg.inv(X.T@X + self.lamb*X.T@A@np.linalg.inv(A.T@A)@A.T@X)@(
                X.T@y + self.lamb*X.T@A@np.linalg.inv(A.T@A)@A.T@y)

        self.is_fitted_ = True
        return self
    
def est_anchor(x_list, y_list, true_para):
	xs, ys, anchors = [], [], []
	for i in range(len(x_list)):
		xs.append(x_list[i])
		ys.append(y_list[i])
		onehot = np.zeros(len(x_list)-1)
		if i + 1 < len(x_list):
			onehot[i] = 1
		anchors.append([onehot] * np.shape(x_list[i])[0])
	
	X, y, A = np.concatenate(xs, 0), np.squeeze(np.concatenate(ys, 0)), np.concatenate(anchors, 0)
	error_min = 1e9
	beta = 0

	for reg in [0, 1, 2, 4, 8, 10, 15, 20, 30, 40, 60, 80, 90, 100, 150, 200, 500, 1000, 5000, 10000]:
		model = AnchorRegression(lamb=reg)
		model.fit(X, y, A)
		cand = np.squeeze(model.coef_)
		# cov_x = sum([np.matmul(x.T, x) / np.shape(x)[0] for x in x_list]) / len(x_list)
		error = np.linalg.norm(cand - true_para)
		if error < error_min:
			error_min = error
			beta = cand

	return beta
    

def est_drig(data, gamma, y_idx=-1, del_idx=None, unif_weight=False):
    """DRIG estimator.

    Args:
        data (list of numpy arrays): a list of data from all environments, where the first element is the observational environment.
        gamma (float): hyperparameter in DRIG.
        y_idx (int, optional): index of the response variable. Defaults to -1.
        del_idx (int, optional): index of the variable to exclude. Defaults to None.
        unif_weight (bool, optional): whether to use uniform weights. Defaults to False.

    Returns:
        numpy array: estimated coefficients.
    """
    if del_idx is None:
        del_idx = y_idx
    ## number of environment
    m = len(data)
    if unif_weight:
        w = [1/m]*m
    else:
        w = [data[e].shape[0] for e in range(m)]
        w = [a/sum(w) for a in w]
    ## gram matrices
    gram_x = [] ## E[XX^T]
    gram_xy = [] ## E[XY]
    for e in range(m):
        data_e = data[e]
        n = data_e.shape[0]
        y = data_e[:, y_idx]
        x = np.delete(data_e, (y_idx, del_idx), 1)
        gram_x.append(x.T.dot(x)/n)
        gram_xy.append(x.T.dot(y)/n)
    G = (1 - gamma)*gram_x[0] + gamma*sum([a*b for a,b in zip(gram_x, w)])
    Z = (1 - gamma)*gram_xy[0] + gamma*sum([a*b for a,b in zip(gram_xy, w)])
    return np.linalg.inv(G).dot(Z)

