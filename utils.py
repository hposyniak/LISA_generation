import sys
import os
import pathlib
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import pandas as pd
from collections.abc import Iterator
from pytorch_minimize.optim import MinimizeWrapper # this is intended for full batch trainnig, can probably simplify and do without
from itertools import combinations
import random # added by helena during debugging

shuffles = []
GLOBAL_SEED = random.randint(0, 2**64-1)

def generate_normal_time_series_set(p: int, num_series: int, noise_amp: float, t_init: float,
                                    t_end: float, seed: int=GLOBAL_SEED) -> tuple:
	"""
	:param p: How many elements to generate in each sequence.
	:param num_series: How many series to generate.
	:param noise_amp: The standard deviation of the ditribution from which elements are drawn.
	:param t_init: The first time point.
	:param t_end: The last time point.
	:param seed: The seed used for ransomisation.
	:return X: A collection of time series.
	:return T: The time points on which the time series are defined.
	Generate a normal time series data set where each of the elements is drawn from a normal
	distribution centered at zero.
	"""
	torch.manual_seed(seed)
	X = torch.normal(0, noise_amp, (num_series, p))
	T = torch.linspace(t_init, t_end, p)
	return X, T

def generate_anomalous_time_series_set(p: int, num_series: int, noise_amp: float, spike_amp: float,
                                       max_duration: int, t_init: float, t_end: float,
                                       seed: int=GLOBAL_SEED) -> tuple:
	"""
	:param p: How many elements to generate in each sequence.
	:param num_series: How many series to generate.
	:param noise_amp: The standard deviation of the ditribution from which elements are drawn.
	:param spike_amp: The standard deviation of the ditribution from which anomolous elements are
                          drawn.
	:param max_duration: The maximum duration of an anomalous spike.
	:param t_init: The first time point.
	:param t_end: The last time point.
	:param seed: The seed used for ransomisation.
	:return Y: A collection of anomalous time series.
	:return T: The time points on which the time series are defined.
	Generate an anomalous time series data set where the elements of each sequence are from a normal
        distribution centered at zero. Then, insert anomalous spikes of random amplitudes and durations.
	"""
	torch.manual_seed(seed)
	Y = torch.normal(0, noise_amp, (num_series, p))
	for y in Y:
		# Allowing between five and ten spikes.
		spike_num = torch.randint(low=5, high=10, size=())
		durations = torch.randint(low=1, high=max_duration, size=(spike_num,))
		spike_start_idxs = torch.randperm(p-max_duration)[:spike_num]
		for start_idx, duration in zip(spike_start_idxs, durations):
			y[start_idx:start_idx+duration] += torch.normal(0.0, spike_amp, (duration,))
	T = torch.linspace(t_init, t_end, p)
	return Y, T

def plot_behaviour(X_norm, T_norm, Y_anom, T_anom):
	"""
	:param X_norm: A normally behaving time series.
	:param T_norm: The time points upon which the X_norm is defined.
	:param X_anom: A anomalous time series.
	:param T_anom: The time points upon which the Y_anom is defined.
	Overlay an anomalous time series on a normally behaving one.
	"""
	plt.figure()
	plt.plot(T_norm, X_norm[0], label="Normal")
	plt.plot(T_anom, Y_anom[1], label="Anomalous")
	plt.ylabel("$y(t)$")
	plt.xlabel("t")
	plt.grid()
	leg = plt.legend()
	plt.show()

def make_atomised_training_set(X: torch.Tensor, T: torch.Tensor) -> list:
	"""
	:param X: A time series.
	:param T: The time points upon which X is defined.
	:return atomised: A list of tuples containing training data points.
	Convert input time series data into atomised tuple chunks.
	"""
	X_flat = torch.flatten(X)
	T_flat = T.repeat(X.size()[0])
	atomised = [(xt, t) for xt, t in zip(X_flat, T_flat)]
	return atomised

class DataGetter:
	"""
	A pickleable mock-up of a Python iterator on a torch.utils.Dataloader. A regular
        DataLoader is sufficient but this structure is used to facilitate the use of the
        covalent package.
	"""

	def __init__(self, X: torch.Tensor, batch_size: int, auto_shuffle, seed: int=GLOBAL_SEED) -> None:
		"""
		:param X: The data set.
		:param batch_size: How many training data to use per parameter update.
		:param seed: The seed used for ransomisation.
		Calls the _init_data method on intialization of a DataGetter object.
		"""
		torch.manual_seed(seed)
		self.X = X
		self.batch_size = batch_size
		self.auto_shuffle=auto_shuffle
		self.data = []
		self._init_data(iter(torch.utils.data.DataLoader(self.X, batch_size=self.batch_size, shuffle=self.auto_shuffle)))  # shuffle being true here (and below) just means the data is shuffled for iterating through a single time, set to false here and below to save the batches for glitch analysis

	def _init_data(self, iterator: Iterator) -> None:
		"""
		:param iterator:
		Calls an iterator to add data to a list.
		"""
		if not self.auto_shuffle:
			self._shuffle_and_save()
		x = next(iterator, None)  # not the next defined below
		while x is not None:
			if x.shape[0] == self.batch_size:
				self.data.append(x)
			x = next(iterator, None)

	def _shuffle_and_save(self):
		perm = np.random.permutation(self.X.shape[0])
		for indx in perm:
			shuffles.append(indx)
		self.X = self.X[perm]
		shuffles_save_path = "redo_unfiltered_4_series_indxs.pkl"
		pathlib.Path(os.path.abspath(shuffles_save_path)).parent.mkdir(parents=True, exist_ok=True)
		pickle.dump(shuffles, open(os.path.abspath(shuffles_save_path), "wb"))

	def __next__(self) -> tuple:
		"""
		:return next_frame: The next data point.
		Defines behaviour analogous to that of the native Python next() method by calling the pop()
                method of the data attribute.
		"""
		try:
			next_frame = self.data.pop()
			return next_frame
		except IndexError:  # Catch when the data set runs out of elements.
			self._init_data(iter(torch.utils.data.DataLoader(self.X, batch_size=self.batch_size, shuffle=self.auto_shuffle)))  # !! no control over reuse of training examples if True
			next_frame = self.data.pop()
			return next_frame

def get_training_cycler(Xtr: torch.Tensor, batch_size: int, seed: int=GLOBAL_SEED) -> DataGetter:
	"""
	:param Xtr: Training data.
	:param batch_size: How many data points there are in each bach.
	:param seed: The seed used for ransomisation.
	:return data_getter: An instance of the DataGetter class with the specified parameters.
	Gets an instance of the DataGetter class which behaves analogously to the next function with an
        iterator but is pickleable.
	"""
	data_getter = DataGetter(Xtr, batch_size, seed)
	return data_getter

def create_diagonal_circuit(gammas: torch.Tensor, n_qubits: int, k: int=None) -> None:  # !! simple_rewinder assumes this method is called D
	"""
	:param gammas: The diagonal entries of D.
	:param n_qubits: The number of qubits in the circuit.
	:param k: The number of qubits used to approximate D.
	:param get_probs: Whether or not to return the measurement probabilities.
	:return probs: The measurement probabilities of the qubits.
	Generates an n qubit quantum circuit according to a k local Walsh operator expansion.
        Here, k local means that at most k of the qubits interact.
	"""
	if k is None:
		k = n_qubits
	cnt = 0
	for i in range(1, k+1):
		for comb in combinations(range(n_qubits), i):
			if len(comb) == 1:
				qml.RZ(gammas[cnt], wires=[comb[0]])
				cnt += 1
			elif len(comb) > 1:
				cnots = [comb[i:i+2] for i in range(len(comb)-1)]
				for j in cnots:
					qml.CNOT(wires=j)
				qml.RZ(gammas[cnt], wires=[comb[-1]])
				cnt += 1
				for j in cnots[::-1]:
					qml.CNOT(wires=j)

class Optimiser:
	"""
	A torch based way of performing parameter updates.
	"""

	def __init__(self, function: callable, variables: dict, optimiser_parameters: dict):

		"""
		:param function: The function which will compute loss.
		:param variables: The arguments for the loss function.
		:param optimiser_parameters: A set of specifications for optimisation.
		Initialises class variables.
		"""
		self.function = function
		self.variables = variables
		self.optimiser_parameters = optimiser_parameters
		self.optimiser = MinimizeWrapper(list(self.variables.values()),
                                                      self.optimiser_parameters)
		self.loss_iterations = []

	def optimise(self):
		"""
		Performs optimisation steps.
		"""
		def closure():
			"""
			Computes loss.
			"""
			print("closure")
			self.optimiser.zero_grad()
			print("ndbh")
			loss = self.function(**self.variables)
			print("loss")
			self.loss_iterations.append(float(loss))
			print("loss appended")
			loss.backward()
			#("start")
			#print(self.variables)
			#print("end")
			#print()
			
			return loss
		self.optimiser.step(closure=closure)
		return self.variables
	
	def get_loss_iterations(self):
		"""
		Gets the loss history.
		"""
		return self.loss_iterations

def load_data_from_file(path):
	"""
	:param path: The path of the data.
	:return data: The fetched data.
	load data from a pickle file.
	"""
	data = torch.tensor(pd.read_pickle(path))
	data = np.expand_dims(data, axis=1)
	return data

def save_to_file(data, path, copy=0):
	if copy > 100:
		print("over 100 models are saved, consider deleting some")
	if os.path.isfile(path):
		name, extension = os.path.splitext(path)
		if copy > 0:
			print("{} already exists, attempting to save as {}".format(path, "".join([name[:-len(str(copy))], str(copy+1), extension])))
			save_to_file(data, "".join([name[:-len(str(copy))], str(copy+1), extension]), copy+1)
		else:
			print("{} already exists, attempting to save as {}".format(path, "".join([name, str(copy+1), extension])))
			save_to_file(data, "".join([name, str(copy+1), extension]), copy+1)
	else:
		pathlib.Path(os.path.abspath(path)).parent.mkdir(parents=True, exist_ok=True)
		pickle.dump(data, open(os.path.abspath(path), "wb"))

def rescale(data: torch.tensor):
	epsillon = 0.01  # A value to distinguish negative pi from positive pi.
	newMin = -torch.pi+epsillon
	newMax = torch.pi-epsillon
	for seriesNo in range(data.shape[2]):
		series = data[:, 0, seriesNo]
		oldMin = np.min(series)
		oldMax = np.max(series)
		series = (series-oldMin)/(oldMax-oldMin)*(newMax-newMin)+newMin
		data[:, 0, seriesNo] = series
	return data

def data_sanity_check(data: torch.Tensor, is_complex: bool=True):

	"""
	:param data: Dataset to be fed to the rewinder model.
	:param is_complex: Whether or not the data was generated from a specified number of time series
							and time points.
	Checks if the data of the correct dimensionality.
	"""

	X = np.asarray(data)
	if is_complex and len(X.shape) != 3:
		raise TypeError(' '.join(["The data must be three dimensional, with dimensions",
												"corresponding to the series, features, and time in that",
												"order."]))
	elif not is_complex and len(X.shape) != 2:
		raise TypeError(' '.join(["The data must be two dimensional, with the first",
												"dimension corresponding to the training example number"
												"and the second dimension correspondng to the signal",
												"value and corresponding time, as a tuple."]))
