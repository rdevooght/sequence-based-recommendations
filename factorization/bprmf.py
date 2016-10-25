from __future__ import division
from __future__ import print_function
import numpy as np
import math
import random
import re
import os
import glob
import sys
from time import time

class BPRMF(object):
	''' Implementation of the algorithm presented in "BPR: Bayesian personalized ranking from implicit feedback", by Rendle S. et al., 2009.

	The adaptive sampling algorithm is adapted from "Improving pairwise learning for item recommendation from implicit feedback", by Rendle S. et al., 2014
	'''

	def __init__(self, k = 32, reg = 0.0025, learning_rate = 0.05, annealing=1., init_sigma = 1, adaptive_sampling=True, sampling_bias=500):
		self.name = 'BPRMF'
		self.k = k
		self.reg = reg
		self.learning_rate = learning_rate # self.learning_rate will change due to annealing.
		self.init_learning_rate = learning_rate # self.init_learning_rate keeps the original value (for filename)
		self.annealing_rate = annealing
		self.init_sigma = init_sigma
		self.adaptive_sampling = adaptive_sampling
		self.sampling_bias = sampling_bias # lambda parameter in "Improving pairwise learning for item recommendation from implicit feedback", by Rendle S. et al., 2014
		self.max_length = np.inf # For compatibility with the RNNs

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "bprmf_ne"+str(epochs)+"_lr"+str(self.init_learning_rate)+"_an"+str(self.annealing_rate)+"_k"+str(self.k)+"_reg"+str(self.reg)+"_ini"+str(self.init_sigma)
		if self.adaptive_sampling:
			filename += "_as"+str(self.sampling_bias)
		return filename+".npz"

	def init_model(self):
		''' Initialize the model parameters
		'''
		self.V = self.init_sigma * np.random.randn(self.n_users, self.k).astype(np.float32)
		self.H = self.init_sigma * np.random.randn(self.n_items, self.k).astype(np.float32)
		self.bias = np.zeros(self.n_items).astype(np.float32)

	def sgd_step(self, user, true_item, false_item):
		''' Make one SGD update, given that the interaction between user and true_item exists, 
		but the one between user and false_item does not.
		user, true_item and false_item are all user or item ids.

		return error
		'''

		# Compute error
		x_true = self.bias[true_item] + np.dot(self.V[user, :], self.H[true_item, :]) 
		x_false = self.bias[false_item] + np.dot(self.V[user, :], self.H[false_item, :]) 
		delta = 1 - 1 / (1 + math.exp(min(10, max(-10, x_false - x_true)))) # Bound x_true - x_false in [-10, 10] to avoid overflow
		
		# Update CF
		self.V[user, :] += self.learning_rate * ( delta * (self.H[true_item, :] - self.H[false_item, :]) - self.reg * self.V[user, :])
		self.H[true_item, :] += self.learning_rate * ( delta * self.V[user, :] - self.reg * self.H[true_item, :])
		self.H[false_item, :] += self.learning_rate * ( -delta * self.V[user, :] - self.reg / 10 * self.H[false_item, :])
		self.bias[true_item] += self.learning_rate * (delta - self.reg * self.bias[true_item])
		self.bias[false_item] += self.learning_rate * (- delta - self.reg * self.bias[false_item])

		return delta

	def prepare_model(self, dataset):
		'''Must be called before using train, load or top_k_recommendations
		'''
		self.n_items = dataset.n_items
		self.n_users = dataset.n_users

	def change_data_format(self, dataset):
		'''Gets a generator of data in the sequence format and save data in two formats:
		a list of (user, item) iteractions pairs for sampling
		a dictionary user_id => [list of items] for fast access to all the items rated by any user.
		'''
		# self.interactions_group_by_users = {}
		# for sequence, user_id in dataset.training_set(epochs=1):
		# 	self.interactions_group_by_users[int(user_id)] = [s[0] for s in sequence]

		self.users = np.zeros((self.n_users,2), dtype=np.int32)
		self.items = np.zeros(dataset.training_set.n_interactions, dtype=np.int32)
		cursor = 0
		with open(dataset.training_set.filename, 'r') as f:
			for sequence in f:
				sequence = sequence.split()
				items = map(int, sequence[1::2])
				self.users[int(sequence[0]), :] = [cursor, len(items)]
				self.items[cursor:cursor+len(items)] = items
				cursor += len(items)

	def compute_factor_rankings(self):
		'''Rank items according to each factor in order to do adaptive sampling
		'''

		self.ranks = np.argsort(self.H, axis=0)
		self.var = np.var(self.H, axis=0)

	def get_training_sample(self):
		'''Pick a random triplet from self.triplets and a random false next item.
		returns a tuple of ids : (user, true_item, false_item)
		'''

		user_id = random.randrange(self.n_users)
		while self.users[user_id,1] < 2:
			user_id = random.randrange(self.n_users)
		user_items = self.items[self.users[user_id,0]:self.users[user_id,0]+self.users[user_id,1]]
		true_item = random.choice(user_items)
		if self.adaptive_sampling:
			while True:
				rank = np.random.exponential(scale=self.sampling_bias)
				while rank >= self.n_items:
					rank = np.random.exponential(scale=self.sampling_bias)
				factor_signs = np.sign(self.V[user_id, :])
				factor_prob = np.abs(self.V[user_id, :]) * self.var
				f = np.random.choice(self.k, p=factor_prob/sum(factor_prob))
				false_item = self.ranks[int(rank) * factor_signs[f],f]
				if false_item not in user_items:
					break
		else:
			false_item = random.randrange(self.n_items)
			while false_item in user_items:
				false_item = random.randrange(self.n_items)

		return (user_id, true_item, false_item)

	def top_k_recommendations(self, sequence, user_id=None, k=10, exclude=None):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		if exclude is None:
			exclude = []

		last_item = sequence[-1][0]
		output = self.bias + np.dot(self.V[user_id, :], self.H.T)

		# Put low similarity to viewed items to exclude them from recommendations
		output[[i[0] for i in sequence]] = -np.inf
		output[exclude] = -np.inf

		# find top k according to output
		return list(np.argpartition(-output, range(k))[:k])

	def train(self, dataset, 
		max_time=np.inf, 
		progress=2.0, 
		time_based_progress=False, 
		autosave='All', 
		save_dir='', 
		min_iterations=0, 
		max_iter=np.inf, 
		max_progress_interval=np.inf,
		load_last_model=False,
		early_stopping=None):
		'''Train the model based on the sequence given by the training_set

		max_time is used to set the maximumn amount of time (in seconds) that the training can last before being stop.
			By default, max_time=np.inf, which means that the training will last until the training_set runs out, or the user interrupt the program.
		
		progress is used to set when progress information should be printed during training. It can be either an int or a float:
			integer : print at linear intervals specified by the value of progress (i.e. : progress, 2*progress, 3*progress, ...)
			float : print at geometric intervals specified by the value of progress (i.e. : progress, progress^2, progress^3, ...)

		max_progress_interval can be used to have geometric intervals in the begining then switch to linear intervals. 
			It ensures, independently of the progress parameter, that progress is shown at least every max_progress_interval.

		time_based_progress is used to choose between using number of iterations or time as a progress indicator. True means time (in seconds) is used, False means number of iterations.

		autosave is used to set whether the model should be saved during training. It can take several values:
			All : the model will be saved each time progress info is printed.
			Best : save only the best model so far
			None : does not save

		min_iterations is used to set a minimum of iterations before printing the first information (and saving the model).

		save_dir is the path to the directory where models are saved.

		load_last_model: if true, find the latest model in the directory where models should be saved, and load it before starting training.

		early_stopping : should be a callable that will recieve the list of validation error and the corresponding epochs and return a boolean indicating whether the learning should stop.
		'''

		# Change data format
		self.change_data_format(dataset)

		# Load last model if needed, else initialise the model
		iterations = 0
		epochs_offset = 0
		if load_last_model:
			epochs_offset = self.load_last(save_dir)
		if epochs_offset == 0:
			self.init_model()

		start_time = time()
		next_save = int(progress)
		val_costs = []
		train_costs = []
		current_train_cost = []
		epochs = []
		while (time() - start_time < max_time and iterations < max_iter):

			if self.adaptive_sampling and iterations%int(self.n_items * np.log(self.n_items)) == 0:
				self.compute_factor_rankings()

			# Train with a new batch
			cost = self.sgd_step(*self.get_training_sample())

			current_train_cost.append(cost)

			# Cool learning rate
			if iterations % dataset.training_set.n_interactions == 0:
				self.learning_rate *= self.annealing_rate

			# Check if it is time to save the model
			iterations += 1

			if time_based_progress:
				progress_indicator = int(time() - start_time)
			else:
				progress_indicator = iterations

			if progress_indicator >= next_save:

				if progress_indicator >= min_iterations:
					
					# Save current epoch
					epochs.append(epochs_offset + iterations / dataset.training_set.n_interactions)

					# Average train cost
					train_costs.append(np.mean(current_train_cost))
					current_train_cost = []

					# Compute validation cost
					costs = []
					for sequence, user_id in dataset.validation_set(epochs=1):
						top_k = self.top_k_recommendations(sequence[:len(sequence)//2], user_id=user_id)
						costs.append(len(set([s[0] for s in sequence[len(sequence)//2:]]) & set(top_k))/len(top_k))
						# costs.append(int(sequence[len(sequence)//2][0] in top_k))
					last_cost = np.mean(costs)
					val_costs.append(last_cost)

					# Print info
					self._print_progress(iterations, epochs[-1], start_time, val_costs, train_costs)

					# Save model
					if autosave == 'All':
						filename = save_dir + self._get_model_filename(round(epochs[-1], 3))
						self.save(filename)
					elif autosave == 'Best':
						if len(val_costs) == 1:
							filename = save_dir + self._get_model_filename(round(epochs[-1], 3))
							self.save(filename)
						elif val_costs[-1] > max(val_costs[:-1]):
							try:
								os.remove(filename)
							except OSError:
								print('Warning : Previous model could not be deleted')
							filename = save_dir + self._get_model_filename(round(epochs[-1], 3))
							self.save(filename)

					if early_stopping is not None:
						if early_stopping(epochs, val_costs):
							return (max(val_costs), time()-start_time, filename)


				# Compute next checkpoint
				if isinstance(progress, int):
					next_save += min(progress, max_progress_interval)
				else:
					next_save += min(max_progress_interval, next_save * (progress - 1))

	def _print_progress(self, iterations, epochs, start_time, val_costs, train_costs):
		'''Print learning progress in terminal
		'''
		print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
		print("Last train cost : ", train_costs[-1])
		print("Last val cost : ", val_costs[-1])
		print("Mean val cost : ", np.mean(val_costs))
		print('-----------------')

		# Print on stderr for easier recording of progress
		print(iterations, epochs, time() - start_time, train_costs[-1], val_costs[-1], file=sys.stderr)

	def save(self, filename):
		'''Save the parameters of a network into a file
		'''
		print('Save model in ' + filename)
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))
		np.savez(filename, V=self.V, H=self.H, bias=self.bias)

	def load_last(self, save_dir):
		'''Load last model from dir
		'''
		def extract_number_of_epochs(filename):
			m = re.search('_ne([0-9]+(\.[0-9]+)?)_', filename)
			return float(m.group(1))

		# Get all the models for this RNN
		file = save_dir + self._get_model_filename("*")
		file = np.array(glob.glob(file))

		if len(file) == 0:
			print('No previous model, starting from scratch')
			return 0

		# Find last model and load it
		last_batch = np.amax(np.array(map(extract_number_of_epochs, file)))
		last_model = save_dir + self._get_model_filename(last_batch)
		print('Starting from model ' + last_model)
		self.load(last_model)

		return last_batch
		

	def load(self, filename):
		'''Load parameters values form a file
		'''
		f = np.load(filename)
		self.V = f['V']
		self.H = f['H']
		self.bias = f['bias']