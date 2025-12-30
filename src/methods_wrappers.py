from .pre_methods.cdanzig import causalDantzig
from .pre_methods.others import *
from .pre_methods.eills import brute_force, pooled_least_squares
from .negdro import negDRO
import numpy as np
import torch


##############################################
#
#                Methods
#
##############################################

def erm(x_list, y_list):
    return pooled_least_squares(x_list, y_list)


def oracle_icp(x_list, y_list, beta_star):
    y_list = [y_list[e].reshape(-1, 1) for e in range(len(y_list))]  # Adjusted to len(y_list)
    data_list = []
    for i in range(len(x_list)):
        data_list.append((torch.tensor(x_list[i]).float(), torch.tensor(y_list[i]).float()))
    
    error_min = 1e9
    beta = 0
    for alpha in [0.9, 0.95, 0.99, 0.995]:
        model = InvariantCausalPrediction(data_list, args={"alpha": alpha, "verbose": False})
        cand = np.squeeze(model.solution().numpy())
        error = np.linalg.norm(cand - beta_star)
        if error < error_min:
            error_min = error
            beta = cand
    
    return beta

def oracle_anchor(x_list, y_list, beta_star):
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
		error = np.sum(np.square(cand - beta_star))
		if error < error_min:
			error_min = error
			beta = cand

	return beta

def eills(x_list, y_list):
    y_list = [y_list[e].reshape(-1, 1) for e in range(len(y_list))]  # Adjusted to len(y_list)
    return brute_force(x_list, y_list, 20, loss_type='eills')

def causal_dantzig_reg(x_list, y_list):
    X, y = np.concatenate(x_list, 0), np.squeeze(np.concatenate(y_list, 0))
    Inds = []
    for i in range(len(x_list)):
        Inds.append(np.array([i] * x_list[i].shape[0]))
    ExpInd = np.squeeze(np.concatenate(Inds, 0))
    
    return causalDantzig(X, y, ExpInd)

def oracle_drig(x_list, y_list, beta_star):
    xy_list = [np.hstack((x_list[e], y_list[e].reshape(-1,1))) for e in range(len(y_list))]
    error_min = 1e9
    beta = 0
    for reg in [0, 1, 2, 4, 8, 10, 15, 20, 30, 40, 60, 80, 90, 100, 150, 200, 500, 1000, 5000, 10000]:
        cand = est_drig(xy_list, gamma = reg)
        error = np.linalg.norm(cand - beta_star)
        if error < error_min:
            error_min = error
            beta = cand
    return beta

def drig(x_list, y_list, gamma):
    xy_list = [np.hstack((x_list[e], y_list[e].reshape(-1,1))) for e in range(len(y_list))]
    beta = est_drig(xy_list, gamma)
    return beta

def causal_dantzig(x_list, y_list):
	n0 = np.shape(x_list[0])[0]
	n1 = np.shape(x_list[1])[0]
	z = np.matmul(x_list[0].T, y_list[0]) / n0 - np.matmul(x_list[1].T, y_list[1]) / n1
	g = np.matmul(x_list[0].T, x_list[0]) / n0 - np.matmul(x_list[1].T, x_list[1]) / n1
	return np.squeeze(np.matmul(np.linalg.inv(g), z))

def negdro(x_list, y_list, gamma):
    b_neg, _ = negDRO(x_list, y_list, gamma=gamma, early_stop=True, num_iter=1500)
    return b_neg
