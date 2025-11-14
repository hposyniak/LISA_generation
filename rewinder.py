import sys
import os
import argparse
import math
import pickle
import time

import torch
import numpy as np
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

import utils

#torch.set_default_dtype(torch.DoubleTensor)
torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_dtype(torch.float64)
n_qubits = 2
dev = qml.device("lightning.qubit", wires=n_qubits, shots=256)


time_indexes = []

def get_series_training_cycler(Xtr: np.array, n_series_batch: int) -> utils.DataGetter:
	"""
	:param Xtr: Training data consisting of time series', features, and time in the
                    dimensions.
	:param n_series_batch: How many series to fetch for the batch.
	:return x_cycler: An object which iterates through time series'.
	Get a cycler to iterate over time series' which are randomly selected.
	"""
	x_cycler = utils.DataGetter(Xtr, n_series_batch, auto_shuffle=False)
	return x_cycler

def get_timepoint_training_cycler(Xtr: np.array, n_t_batch: int) -> utils.DataGetter:
	"""
	:param Xtr: Training data consisting of time series', features, and time in the
                    dimensions.
	:param n_t_batch: How many time points to fetch for the batch.
	:return t_cycler: An object which iterates through time points.
	Get a cycler to iterate through time points which are randomly selected.
	"""
	n_time_points = Xtr.shape[2]
	T = torch.tensor(np.arange(n_time_points))
	t_cycler = utils.DataGetter(T, n_t_batch, auto_shuffle=True)
	return t_cycler

@qml.qnode(dev, interface="torch")
def get_anomaly_expec(x: np.array, t: float, D: torch.tensor, alpha: torch.tensor,
                      wires: qml.wires.Wires, k: int, embed_func: callable,
                      transform_func: callable, diag_func: callable, observable: list,
                      embed_func_params: dict={}, transform_func_params: dict={}) -> float:
	"""
	:param x: The value in teach dimension of a data point of a series at time t.
	:param t: The time corresponding to the data point x.
	:param D: Diagonal entries of D.
	:param alpha: Parameters of W with the weights for each layer, each qubit, and each rotation
                      on individual dimensions in that order.
	:param wires: The qubits to apply the circuit to.import sys
	:param k: The number of qubits used to approximate D.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param diag_func: The function which approximates D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:param embed_func_params: Optional additional parameters for embedding.
	:param transform_func_params: Optional additional parameters for rewinding.
	:return expval: The expected value of the circuit output.
	Simulate the circuit with a single data point and calculate the expected value.
	"""
	print(x)
	#sys.exit(2)
	embed_func(x, wires=wires, **embed_func_params)
	transform_func(alpha, wires, **transform_func_params)
	diag_func(D*t, n_qubits, k=k)
	qml.adjoint(transform_func)(alpha, wires=range(n_qubits), **transform_func_params)
	coeffs = np.ones(len(wires))/len(wires)
	H = qml.Hamiltonian(coeffs, observable)
	expval = qml.expval(H)
	return expval

def get_single_point_cost(x: np.array, t: float, alphas: torch.tensor, eta_0: float,
                          M_sample_func: callable, sigmas: float, mus: float, N_E: int, wires: int,
                          k: int, embed_func: callable, transform_func: callable,
                          diag_func: callable, observable: list, embed_func_params: dict={},
                          transform_func_params: dict={}) -> float:
	"""
	:param x: The value in teach dimension of a data point of a series at time t.
	:param t: The time corresponding to the data point x.
	:param alphas: Parameters of W with the weights for each layer, each qubit, and each
                       rotation on individual dimensions in that order.
	:param eta_0: The learnable centre of the cluster of the expected values.
	:param M_sample_func: The function descibing the distribution from which D variations are
                              sampled.
	:param sigmas: The standard deviations of the normal distributions from which elements of D
                      are sampled.
	:param mus: The means of the normal distributions from which elements of D matricies are
                    sampled.
	:param N_E: How many D matrices to sample.
	:param wires: The number of qubits.
	:param k: The number of qubits used to approximate D.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param diag_func: The function which approximates D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:param embed_func_params: Optional additional parameters for embedding.
	:param transform_func_params: Optional additional parameters for rewinding.
	:return single_point_cose: The loss over a set of sampled rewinding Hamiltonians.
	Calculate the loss over a set of rewinding Hamiltonians for a single time point of a single time
        series.
	"""
	expecs = torch.zeros(N_E)
	for i in range(N_E):
		D = M_sample_func(sigmas, mus)
		expec = get_anomaly_expec(x, t, D, alphas, wires, k, embed_func, transform_func,
                                          diag_func, observable, embed_func_params={},
                                          transform_func_params={})
		expecs[i] = expec
	mean = expecs.mean()
	single_point_cost = (eta_0-mean)**2  # Excluding the factor 1/4.
	return single_point_cost

def sample_M(sigma: float, mus: float):
	"""
	:param sigma: The standard deviation of the normal distribution which is sampled from.
	:param mu: The mean of thenormal distribution which is sampled from.
	:return D: The sampled value.
	Sample values from a normal distribution.
	"""
	D = torch.normal(mus, sigma.abs())
	return D

def get_time_series_cost(xt: np.array, alphas: torch.tensor, eta_0: float, M_sample_func: callable,
                         sigmas: torch.tensor, mus: torch.tensor, N_E: int, wires: int, k: int,
                         embed_func: callable, transform_func: callable, diag_func: callable,
                         observable: list, t_cycler: utils.DataGetter, embed_func_params: dict={},
                         transform_func_params: dict={}, start: float=0.1,
                         end: float=2*np.pi) -> float:
	"""
	:paramm xt: Values in each dimension across all measured time points of a single time
                    series.
	:param alphas: Parameters of W with the weights for each layer, each qubit, and each
                      rotation on individual dimensions in that order.
	:param eta_0: The learnable centre of the cluster of the expected values.
	:param M_sample_func: The function descibing the distribution from which D variations are
                              sampled.
	:param sigmas: The standard deviations of the normal distributions from which elements of
                      the D matricies are sampled.
  	:param mus: The means of the normal distributions from which elements of the D matrices are
                    sampled.
	:param N_E: How many D matrices to sample.
	:param wires: The number of qubits.
	:param k: The number of qubits used to approximate D.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param diag_func: The function which approximates D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:param t_cycler: An object to iterate through time points.
	:param embed_func_params: Optional additional parameters for embedding.
	:param transform_func_params: Optional additional parameters for rewinding.
	:param start: The smallest time point after rescaling.
	:param end: The largest time point after rescaling.
	:return single_time_series_cost: The loss over the data at all sampled time points for one
                                         time series.
	Cycle through a time series with random time points after rescaling them to fit within zero
	and 2*pi, which is chosen because increasing energyvalues by more than 2pi is the same as
	resetting them to zero.
	"""
	if t_cycler is None:  # Evenly space the time points if no cycler is provided, like for testing.
		t_batch = np.arange(xt.shape[1])
		t_scaled = np.linspace(start, end, xt.shape[1], endpoint=True)
	else:  # Randomly sample the time points.
		t_batch = next(t_cycler)
		time_indexes.append(t_batch)
		t_scaled = np.linspace(start, end, xt.shape[1], endpoint=True)[t_batch]
	xt_batch = xt[:, t_batch]
	xfunct = zip(xt_batch.T, t_scaled)
	a_func_t = [get_single_point_cost(x, t, alphas, eta_0, M_sample_func, sigmas, mus, N_E, wires,
                                          k, embed_func, transform_func, diag_func, observable) for
                                          x, t in xfunct]
	single_time_series_cost = torch.tensor(a_func_t, requires_grad=True).mean()
	return single_time_series_cost

def get_loss(alphas: torch.tensor, eta_0: float, M_sample_func: callable, sigmas: torch.tensor,
             mus: torch.tensor, N_E: int, wires: int, k: int, embed_func: callable,
             transform_func: callable, diag_func: callable, observable: list,
             t_cycler: utils.DataGetter, x_cycler: utils.DataGetter, penalty: callable,
             taus: torch.tensor, embed_func_params: dict={},
             transform_func_params: dict={}) -> float:
	"""
	:param alphas: Parameters of W with the weights for each layer, each qubit, and each rotation
                      von individual dimensions in that order.
	:param eta_0: The learnable centre of the cluster of the expected values.
	:param M_sample_func: The function descibing the distribution from which D variations are
                              sampled.
	:param sigmas: The standard deviation of the normal distribution from which elements of D
                      are sampled.
	:param mus: The means of the normal distributions from which elements of D matricies are
                    sampled.
	:param N_E: How many D matrices to sample.
	:param wires: The qubits to apply the circuit to.
	:param k: The number of qubits used to approximate D.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param diag_func: The function which approximates D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:param t_cycler: An object to iterate through time points.
	:param x_cycler: An object to iterate over time series'.
	:param penalty: A function to penalise large sigmas.
	:param taus: Contraction hyperparameters for controlling regularisation.
	:return loss: The loss incurred by a batch of input data.
	Calculate the loss for a batch of input data by accumulating the losses of each time series. The
        number of function evaluations in the optimisation procedure referes to this function.
	"""
	X_batch = next(x_cycler)
	single_costs = torch.zeros(X_batch.shape[0])
	for i, xt in enumerate(X_batch):
		single_time_series_cost = get_time_series_cost(xt, alphas, eta_0, M_sample_func,
                                                               sigmas, mus, N_E, wires, k,
                                                               embed_func, transform_func,
                                                               diag_func, observable, t_cycler,
                                                               embed_func_params={},
                                                               transform_func_params={})
		single_costs[i] = single_time_series_cost
	loss = single_costs.mean()+penalty(sigmas, taus)  # removed the 0.5 multiplying factor.
	return loss

def arctan_penalty(sigmas: torch.tensor, contraction_hyperparameters: torch.tensor) -> float:
	"""
	:param sigmas: The standard deviation of the normal distribution from which elements of D
                      are sampled.
	:param contraction_hyerparameters: Contraction hyperparameters for controlling
                                           regularisation, which must be a single value or match
                                           the shape of the sigmas tensor.
	:param penalty: The extra loss caused by the sigmas.
	Calculate the regularisation penalty based on contraction hyperparameters and sigmas.
	"""
	prefac = 1/(np.pi)
	sum_terms = torch.arctan(2*np.pi*contraction_hyperparameters*torch.abs(sigmas))
	mean = sum_terms.mean()
	penalty = prefac*mean
	return penalty

def train_model(initial_parameters: dict, M_sample_func: callable, N_E: int, wires: int, k: int,
                embed_func: callable, n_qubits: int, transform_func: callable, diag_func: callable,
                observable: list, t_cycler: utils.DataGetter, x_cycler: utils.DataGetter,
                penalty: callable, taus: torch.tensor, optimiser_params: dict) -> dict:
	"""
	:param initial_parameters: Initial values of the trainable parameters.
	:param M_sample_func: The function descibing the distribution from which D variations are
                              sampled.
	:param N_E: How many D matrices to sample.
	:param wires: The qubits to apply the circuit to.
	:param k: The number of qubits used to approximate D.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param n_qubits: The number of qubits composing the circuit.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param diag_func: The function which approximates D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:param t_cycler: An object to iterate through time points.
	:param x_cycler: An object to iterate over time series'.
	:param penalty: A function to penalise large sigmas.
	:param taus: Contraction hyperparameters for controlling regularisation.
	:param optimiser_params: Specifications for the optimisation routine.
	:return results: The optimised paramers and loss history after training.
	Train the model.
	"""
	print("test1")

	alphas = initial_parameters['alphas']
	mus = initial_parameters['mus']
	sigmas = initial_parameters['sigmas']
	eta_0 = initial_parameters['eta_0']
	print("test1.1")

	f = lambda alphas, mus, sigmas, eta_0: get_loss(alphas=alphas, mus=mus, sigmas=sigmas, eta_0=eta_0,
                                                        M_sample_func=M_sample_func, N_E=N_E,
                                                        wires=range(n_qubits), k=k,
                                                        embed_func=embed_func,
                                                        transform_func=transform_func,
                                                        diag_func=diag_func, observable=observable,
                                                        t_cycler=t_cycler, x_cycler=x_cycler,
                                                        penalty=penalty, taus=taus)
	print("test1.2")

	optimiser = utils.Optimiser(f, initial_parameters, optimiser_params)

	print("test1.3")

	opt_params = optimiser.optimise()

	print("test1.4")

	loss_history = optimiser.get_loss_iterations()
	print("test1.5")
	results = {"opt_params": opt_params, "loss_history": loss_history}
	print("test1.6")
	return results

def get_initial_parameters(transform_func: callable, transform_func_layers: int, n_qubits: int,
                           num_distributions: int) -> dict:
	"""
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param transform_func_layers: How many layers to use in the rewinding circuit.
	:param n_qubits: The number of qubits composing the circuit.
	:param num_distributions: The number of elements to sample in D.
	:return init_parameters: Randomly generated initial parameters.
	Generate initial parameters of the circuit to be trained.
	"""
	#alphas_shape = transform_func.shape(transform_func_layers, n_qubits) #!!! # Use this for qml.stronglyEntanglingLayers
	alphas_shape = (transform_func_layers, n_qubits, 3)
	initial_alphas = torch.tensor(np.random.uniform(0, 2*np.pi, size=alphas_shape), requires_grad=True).type(torch.DoubleTensor)
	initial_mus = torch.tensor(np.random.uniform(0, 2*np.pi, num_distributions), requires_grad=True).type(torch.DoubleTensor)
	initial_sigmas = torch.tensor(np.random.uniform(0, 2*np.pi, num_distributions), requires_grad=True).type(torch.DoubleTensor)
	initial_eta_0 = torch.tensor(np.random.uniform(-1, 1), requires_grad=True).type(torch.DoubleTensor)
	init_parameters = {'alphas': initial_alphas, 'mus': initial_mus, 'sigmas': initial_sigmas, 'eta_0': initial_eta_0}
	return init_parameters

def training_workflow(Xtr: torch.tensor, n_series_batch: int, n_t_batch: int, transform_func: callable,
                      n_qubits: int, transform_func_layers: int, embed_func: callable, N_E: int,
                      k: int, observable: list, taus: torch.tensor, optimiser_params: dict,
                      num_distributions: int, penalty: callable, M_sample_func: callable,
                      diag_func: callable) -> dict:
	"""
	:param Xtr_path: Training data.
	:param n_series_batch: How many series to fetch for the batch.
	:param n_t_batch: How many time points to fetch for the batch.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param n_qubits: The number of qubits composing the circuit.
	:param transform_func_layers: How many layers to use in the rewinding circuit.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param N_E: How many D matrices to sample.
	:param k: The number of qubits used to approximate D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:param taus: Contraction hyperparameters for controlling regularisation.
	:param optimiser_params: Specifications for the optimisation routine.
	:param num_distributions: The number of elements to sample in D.
	:param penalty: A function to penalise large sigmas.
	:param M_sample_func: The function descibing the distribution from which D variations are
                        sampled.
	:param diag_func: The function which approximates D.
	:return opt_results: The optimised paramers and loss history after training.
	Load data, package it into training cyclers, prepare the circuit, and
        optimise based on the loss function.
	"""
	utils.data_sanity_check(Xtr)
	print("test2")
	x_cycler = get_series_training_cycler(Xtr, n_series_batch)
	print("test3")
	t_cycler = get_timepoint_training_cycler(Xtr, n_t_batch)
	print("test4")
	init_parameters = get_initial_parameters(transform_func, transform_func_layers, n_qubits,
                                                 num_distributions)
	print("test5")
	opt_results = train_model(init_parameters, M_sample_func, N_E, range(n_qubits), k,
                                  embed_func, n_qubits, transform_func, diag_func, observable,
                                  t_cycler, x_cycler, penalty, taus, optimiser_params)
	print("test6")

	plt.plot(opt_results["loss_history"])
	plt.xlabel("Minibatch Iterations")
	plt.ylabel("Loss")
	plt.grid()
	plt.savefig("loss.png")
	plt.show()
	print("Exiting train_model")
	return opt_results

def get_anomaly_scores(Xte: np.array, penalty: callable, taus: torch.tensor, results_dict: dict,
						M_sample_func: callable, wires: qml.wires.Wires, k: int,
                       embed_func: callable, N_E: int, transform_func: callable,
                       diag_func: callable, observable: list) -> torch.tensor:
	"""
	:param Xte: Testing data consisting of time series', features, and time in the dimensions.
	:param penalty: A function to penalise large sigmas.
	:param taus: Contraction hyperparameters for controlling regularisation.
	:param results_dict: Optimised circuit parameters.
	:param M_sample_func: The function descibing the distribution from which D variations are
                              sampled.
	:param wires: The qubits to apply the circuit to.
	:param k: The number of qubits used to approximate D.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param N_E: How many D matrices to sample.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param diag_func: The function which approximates D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:return scores: The anomaly scores for the input time series'.
	Calculates anomaly scores for each input time series in a data set by comparing the
        anomaly score expected of non-anomalous data to the anomaly scores generated by
        each test time series.
	"""
	opt_params = results_dict['opt_params']
	alphas = opt_params['alphas']
	mus = opt_params['mus']
	sigmas = opt_params['sigmas']
	eta_0 = opt_params['eta_0']
	nonanom_cost = results_dict["loss_history"][len(results_dict["loss_history"])-1]
	scores = torch.zeros(Xte.shape[0])
	for i, yt in enumerate(Xte):
		single_time_series_cost = get_time_series_cost(yt, alphas, eta_0, M_sample_func, sigmas, mus, N_E, wires, k, embed_func, transform_func, diag_func, observable, t_cycler=None)  # No t_cycler implies go through the whole time series
		scores[i] = single_time_series_cost
	return scores

def get_label_prediction_pairs(non_anom_scores: torch.tensor, anom_scores: torch.tensor,
                               threshold: float) -> (list, list):
	"""
	:param non_anom_scores: Anomaly scores of non-anomalous time series' test data.
	:param anom_scores: Anomaly scores of anomalous time series test data.
	:param threshold: The cutoff point for marking a time series as anomalous.
	:return predictions: An assignment of anomalous or non-anomalous for each input time series
                             series, where -1 implies that the corresponding time series is
                             non-anomalous and 1 implies that the corresponding time series is
                             anomalous.
	:return labels: Whether or not each time series is anomalous, where -1 implies that the
                        corresponding time series is non-anomalous and 1 implies that the
                        corresponding time series is anomalous.
	Classify examples as anomalous or non-anomalous by comparing each of their anomaly scores
        to a treshold, and create a set of labels indicating the ground truths of these examples.
	"""
	non_anom_preds = [-1 if score.item() < threshold else 1 for score in non_anom_scores]
	non_anom_labels = [-1 for i in range(len(non_anom_scores))]
	anom_preds = [1 if score.item() >= threshold else -1 for score in anom_scores]
	anom_labels = [1 for i in range(len(anom_scores))]
	labels = []; labels.extend(non_anom_labels); labels.extend(anom_labels)
	preds = []; preds.extend(non_anom_preds); preds.extend(anom_preds)
	return preds, labels

def balanced_acc_score(labels: list, preds: list) -> float:
	"""
	:param labels: Ground truths of input time series', where -1 implies that the
                      corresponding time series is non-anomalous and 1 implies that the
                       corresponding time series is anomalous.
	:param preds: Predicted labels of input time series', where -1 implies that the
                      corresponding time series is non-anomalous and 1 implies that the
                      corresponding time series is anomalous.
	:return acc: The balanced accuracy score.
	Compute the balanced accuracy score of a set of predictions.
	"""
	acc = balanced_accuracy_score(labels, preds)
	return acc

def get_F1_score(labels: list, preds: list) -> float:
	"""
	:param labels: Ground truths of input time series', where -1 implies that the
          			corresponding time series is non-anomalous and 1 implies that the
                      corresponding time series is anomalous.
	:param preds: Predicted labels of input time series', where -1 implies that the
                    corresponding time series is non-anomalous and 1 implies that the
                    corresponding time series is anomalous.
	:return f1: The F1 score.
	Compute the F1 score of a set of predictions.
	"""

	f1 = f1_score(labels, preds)
	return f1

def threshold_scan(non_anom_scores: torch.tensor, anom_scores: torch.tensor,
                   threshold_grid_steps: list) -> (np.array, list, float, float):
	"""
	:param non_anom_scores: Anomaly scores of non-anomalous time series' validation data.
  	:param anom_scores: Anomaly scores of anomalous time series test data.
	:param threshold_grid_steps: Thresholds to check the model performance for.
	:return zetas: Testes threshold values.
	:return accs: Balanced accuracy scores for each tested threshold value.
	:return beta_zeta: The threshold value that resulted in the best balanced accuracy.
	:return best_acc: The best balanced accuracy achieved across the tested thresholds.
	"""
	maxAnomScore, maxNonanomScore = torch.max(anom_scores).item(), torch.max(anom_scores).item()
	maxScore = max(maxAnomScore, maxNonanomScore)
	print("max score ", maxScore)
	zetas = np.linspace(0, maxScore, threshold_grid_steps)
	accs = []
	threshold_grid_step = 0
	for zeta in zetas:
		preds, labels = get_label_prediction_pairs(non_anom_scores, anom_scores, zeta)
		acc = balanced_acc_score(labels, preds)
		accs.append(acc)
		threshold_grid_step += 1
	best_idxs = np.argwhere(accs == np.max(accs)).flatten()
	if best_idxs.shape[0]%2 == 0:  # account for multiple best thresholds by taking the middle one assuming that all indexes in between the lowest and highest which result in the best accuracy also result in the best accuracy
		best_idx = best_idxs[int(best_idxs.shape[0]/2)]
	else:
		best_idx = best_idxs[int((best_idxs.shape[0]-1)/2)]
	best_zeta = zetas[best_idx]
	best_acc = accs[best_idx]
	print("The best evaluated threshold is", best_zeta, "achieving a balanced accuracy of", best_acc)
	print("best zeta: ", best_zeta)
	print("best acc: ", best_acc)
	return best_zeta

def set_threshold(nonanom_anom_scores: torch.tensor, scaling: float) -> float:
	"""
	:param non_anom_scores: Anomaly scores of non-anomalous time series validation data.
	:param scaling: How many standard deviations from the mean of the anomaly scores of the
                        non-anomalous validation set an anomalous score needs to be to classify it
                        as anomalous.
	:return threshold: The value above which the anomaly score of time series' needs to be for
                           them to be classified as anomalous.
	Using the mean and scaled standard deviation of the anomaly scores of a non-anomalous
        validation set, without relying on anomalous data, set the threshold.
	"""
	stats = pd.DataFrame(nonanom_anom_scores.detach().numpy()).describe()
	threshold = list(stats.loc["mean"]+scaling*stats.loc["std"])[0]
	return threshold

def threshold_tuning_workflow(Xte_non_anom: torch.tensor, Xte_anom: str, threshold_grid_steps: list,
                              penalty: callable, taus: torch.tensor, results_dict: dict,
                              M_sample_func: callable, wires: qml.wires.Wires, k: int,
                              embed_func: callable, N_E: int, transform_func: callable,
                              diag_func: callable, observable: list,
                              grid_search: bool) -> (list, torch.tensor, torch.tensor):
	"""
	:param Xte_non_anom: Non-anomalous test data.
	:param Xte_anom: Anomalous test data.
	:param threshold_grid_steps: Thresholds to check the model performance for.
	:param penalty: A function to penalise large sigmas.
	:param taus: Contraction hyperparameters for controlling regularisation.
	:param results_dict: Optimised circuit parameters.
	:param M_sample_func: The function descibing the distribution from which D variations are
                              sampled.
	:param wires: The qubits to apply the circuit to.
	:param k: The number of qubits used to approximate D.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param N_E: How many D matrices to sample.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param diag_func: The function which approximates D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:param grid_serach: Whether or not to tune the threshold by performing a grid serch, where opting
                            not to means it is set using statistical properties of non-anomalous anomaly scores
	:return zeta: The computed threshold.
	:return scores_non_anom: Scores for each non-anomalous validation time series.
	:return scores_anom: Scores for each anomalous validation time series.
	Load validation data, both anomalous and non-anomalous, and compute anomaly scores for
        each of the time series'. Use these anomaly scores to classify each time series an
        anomalous or non-anomalous and then compute the balanced accuracy across various thresholds
        by comparing these classifications to the labels. Select the threshold which resulted in
        the highest performance.
	"""
	scores_non_anom = get_anomaly_scores(Xte_non_anom[:, :, :], penalty, taus, results_dict,
                                             M_sample_func, wires, k, embed_func,
                                             N_E, transform_func, diag_func, observable)  # Here a subset of the time points in a series can be used, for example with [:, :, ::2].
	if grid_search:
		scores_anom = get_anomaly_scores(Xte_anom[:, :, :], penalty, taus, results_dict,
                                                 M_sample_func, wires, k, embed_func, N_E,
                                                 transform_func, diag_func, observable)
		print("anomaly scores for anomalous data     ", scores_anom)
		zeta = threshold_scan(scores_non_anom, scores_anom, threshold_grid_steps)
	else:
		scores_anom = torch.tensor([])
		zeta = set_threshold(scores_non_anom, 2)
	return zeta, scores_non_anom, scores_anom

def testing_workflow(Xte_non_anom: torch.tensor, Xte_anom: torch.tensor, best_zeta: float, penalty: callable,
                     taus: torch.tensor, results_dict: dict, M_sample_func: callable,
                     wires: qml.wires.Wires, k: int, embed_func: callable, N_E: int,
                     transform_func: callable, diag_func: callable,
                     observable: list) -> (float, float, float, float):
	"""
	:param Xte_non_anom: Non-anomalous test data.
	:param Xte_anom: Anomalous test data.
	:param best_zeta: The classification threshold.
	:param penalty: A function to penalise large sigmas.
	:param taus: Contraction hyperparameters for controlling regularisation.
	:param results_dict: Optimised circuit parameters.
	:param M_sample_func: The function descibing the distribution from which D variations are
                              sampled.
	:param wires: The qubits to apply the circuit to.
	:param k: The number of qubits used to approximate D.
	:param embed_func: The function which performs the action of the embedding Hamiltonian.
	:param N_E: How many D matrices to sample.
	:param transform_func: The function which performs the action of the rewinding Hamiltonian.
	:param diag_func: The function which approximates D.
	:param observable: The measurement operator providing a measurement basis specified as an
                           operation for each qubit.
	:return balanced_accuracy: The balanced accuray on the test set.
	:return precision: The precision on the test set.
	:return recall: The recall on the test set.
	:return F1: The F1 on the test set.
	"""
	utils.data_sanity_check(Xte_non_anom)
	utils.data_sanity_check(Xte_anom)
	scores_non_anom = get_anomaly_scores(Xte_non_anom[:, :, :], penalty, taus, results_dict,
                                             M_sample_func, wires, k, embed_func, N_E,
                                             transform_func, diag_func, observable)
	scores_anom = get_anomaly_scores(Xte_anom[:, :, :], penalty, taus, results_dict, M_sample_func,
                                         wires, k, embed_func, N_E, transform_func, diag_func,
                                         observable)
	preds, labels = get_label_prediction_pairs(scores_non_anom, scores_anom, best_zeta)
	balanced_accuracy = balanced_acc_score(labels, preds)
	precision = precision_score(labels, preds)
	recall = recall_score(labels, preds)
	f1 = get_F1_score(labels, preds)
	print("nonanom")
	print(scores_non_anom)
	print("anom")
	print(scores_anom)
	return balanced_accuracy, precision, recall, f1, scores_non_anom, scores_anom

def transform_func(alpha, wires, **transform_func_params):
	for layer in range(alpha.shape[0]):
		for qubit in range(alpha.shape[1]):
			qml.RX(alpha[layer][qubit][0], wires=qubit)
			qml.RY(alpha[layer][qubit][1], wires=qubit)
			qml.RZ(alpha[layer][qubit][2], wires=qubit)
		qml.CNOT(wires=[0, 1])

penalty = arctan_penalty
M_sample_func = sample_M
taus = torch.tensor([15])
embed_func = qml.templates.AngleEmbedding
transform_func = transform_func
diag_func = utils.create_diagonal_circuit
N_E = 10
observable = [qml.PauliZ(i) for i in range(n_qubits)]
x_batch_size = 10
t_batch_size = 10


nonanom_path = "LISA_noise_unfiltered_100.pkl"
anom_path = "LISA_noise_gws_unfiltered_100.pkl"
training_save_path = "results/training_results.pkl"
time_indexes_path = "results/time_indexes.pkl"
noise = torch.from_numpy(utils.rescale(utils.load_data_from_file(nonanom_path)))
print("training")
noise_train = torch.cat((noise[0:70], noise[:70]))
print("test")
training_results = training_workflow(Xtr=noise_train, n_series_batch=x_batch_size, n_t_batch=t_batch_size,
                                     num_distributions=3, transform_func=transform_func, n_qubits=2, 
                                     transform_func_layers=3, embed_func=embed_func, N_E=N_E, k=2,
                                     observable=observable, taus=taus,
                                     optimiser_params={"method": "Powell",
                                                       "options": {"disp": True, "maxfev": 2000, "maxiter": np.inf},
                                                       "jac": False, "bounds": [(0, 2*np.pi) for i in range(24)]+[(-1, 1)]},
                                     penalty=penalty, M_sample_func=M_sample_func, diag_func=diag_func)
print("test2")
utils.save_to_file(training_results, training_save_path)
utils.save_to_file(time_indexes, time_indexes_path)

tuning_save_path = "results/tuning_results.pkl"
training_results = pickle.load(open(training_save_path, "rb"))
events = torch.from_numpy(utils.rescale(utils.load_data_from_file(anom_path)))
print("tuning")
noise_val = noise[70:85]
events_val = events
tuning_results = threshold_tuning_workflow(Xte_non_anom=noise_val, Xte_anom=events_val, threshold_grid_steps=1000,
                                          penalty=penalty, taus=taus, results_dict=training_results,
                                          M_sample_func=M_sample_func, wires=range(n_qubits), k=2,
                                          embed_func=embed_func, N_E=N_E, transform_func=transform_func,
                                          diag_func=diag_func, observable=observable, grid_search=True)
utils.save_to_file(tuning_results, tuning_save_path)

testing_save_path = "results/testing_results.pkl"
print("testing")
noise_test = noise[85:100]
events_test = events
tuning_results = pickle.load(open(tuning_save_path, "rb"))
testing_results = testing_workflow(Xte_non_anom=noise_test, Xte_anom=events_test, penalty=penalty,
                                   best_zeta=tuning_results[0], taus=taus, results_dict=training_results,
                                   M_sample_func=M_sample_func, wires=range(n_qubits), k=2, embed_func=embed_func,
                                   N_E=N_E, transform_func=transform_func, diag_func=diag_func, observable=observable)
utils.save_to_file(testing_results, testing_save_path)
balanced_accuracy, precision, recall, f1, scores_non_anom, scores_anom = testing_results
print("theshold:          ", tuning_results[0])
print("Balanced Accuracy: ", balanced_accuracy)
print("Precision:         ", precision)
print("Recall:            ", recall)
print("F1:                ", f1)





