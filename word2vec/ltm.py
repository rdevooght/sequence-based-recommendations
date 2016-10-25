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
from gensim.models.word2vec import Word2Vec 

class LTM(object):
	""" Implementation of the algorithm proposed in "Latent Trajectory Modeling : A Light and Efficient Way to Introduce Time in Recommender Systems" by Guardia-Sebaoun, E. et al., 2015.
	"""
	def __init__(self, use_trajectory=True, alpha=0.8, k = 32, window = 5, learning_rate=0.025):
		'''

		parameters
		----------
		use_trajectory: boolean
			If True, the original LTM algorithm is used. 
			Else the users features are not computed, and the predictions are made only by taking the items with the closest word2vec representation from the (window/2) last item in the sequence.
		alpha: float in (0,1)
			temporal damping parameter from "Apprentissage de trajectoires temporelles pour la recommandation dans les communautes d'utilisateurs", by Guardia-Sebaoun, E. et al.
		k : int > 0
			number of dimension for the word2vec embedding
		window : int > 0
			window size for the word2vec embedding
		learning_rate: float
			initial learning rate for word2vec. (alpha parameter in the gensim implementation of word2vec)
		'''
		super(LTM, self).__init__()
		self.use_trajectory = use_trajectory
		self.alpha = alpha
		self.k = k
		self.window = window
		self.learning_rate = learning_rate

		self.name = 'Latent Trajectory Modeling'
		self.max_length = np.inf # For compatibility with the RNNs


	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "ltm_ne"+str(epochs)+"_lr"+str(self.learning_rate)+"_k"+str(self.k)+"_w"+str(self.window)
		if self.use_trajectory:
			filename += "_ut"+str(self.alpha)
		return filename

	def user_features(self, sequence):
		'''Compute the transition features of the users based on his sequence of items.
		'''
		features = np.zeros(self.k)
		for i in range(1,len(sequence)):
			features = self.alpha * features + (1 - self.alpha) * (self.w2v_model[str(sequence[i][0])] - self.w2v_model[str(sequence[i-1][0])])

		return features

	def prepare_model(self, dataset):
		''' For compatibility with other methods.
		'''
		pass

	def top_k_recommendations(self, sequence, user_id=None, k=10, exclude=None):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''
		
		if exclude is None:
			exclude = []

		if self.use_trajectory:
			f = self.user_features(sequence)
		else:
			f = np.mean(np.array([self.w2v_model[str(sequence[-i-1][0])] for i in range(self.window//2)]), axis=0) # average over last window/2 items

		top = self.w2v_model.similar_by_vector(f, topn=k+len(sequence)+len(exclude))
		top = [int(i[0]) for i in top if int(i[0]) not in exclude]
		s = [i[0] for i in sequence]
		top = [i for i in top if i not in s]
		return top[:k]

		# # f = f / np.sqrt(np.sum(np.square(f)))
		# # dist = np.dot(self.w2v_model.syn0, f)
		# dist = -np.dot(self.w2v_model.syn0, f) / np.sum(np.square(self.w2v_model.syn0), axis=-1)
		# # dist = np.sum(np.square(self.w2v_model.syn0 - f), axis=-1)

		# # Put low similarity to viewed items to exclude them from recommendations
		# dist[[self.w2v_model.vocab[str(i)].index for i in exclude]] = np.inf
		# dist[[self.w2v_model.vocab[str(i[0])].index for i in sequence]] = np.inf

		# # find top k according to dist
		# return [int(self.w2v_model.index2word[i]) for i in list(np.argpartition(dist, range(k))[:k])]
	
	def word2vec_training_generator(self, dataset):
		'''Take a generator of sequences and produce a generator in the format used by gensim word2vec module
		'''
		for sequence, user_id in dataset.training_set(epochs=1):
			yield [str(i[0]) for i in sequence]

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

		!!!! Contrary to what the train function of other algorithms, here an iteration is equivalent to one epoch !!!!!!!

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

		# Load last model if needed, else initialise the model
		iterations = 0
		epochs_offset = 0
		if load_last_model:
			epochs_offset = self.load_last(save_dir)
		if not hasattr(self, 'w2v_model'):
			self.w2v_model = Word2Vec(iter = 1, min_count = 1, size=self.k, window=self.window, alpha=self.learning_rate, sg=0)
			self.w2v_model.build_vocab([map(str, range(dataset.n_items))])

		# raise ValueError

		start_time = time()
		next_save = int(progress)
		val_costs = []
		epochs = []
		while (time() - start_time < max_time and iterations < max_iter):

			# Train one epoch
			self.w2v_model.train(self.word2vec_training_generator(dataset))

			# Check if it is time to save the model
			iterations += 1

			if time_based_progress:
				progress_indicator = int(time() - start_time)
			else:
				progress_indicator = iterations

			if progress_indicator >= next_save:

				if progress_indicator >= min_iterations:
					
					# Save current epoch
					epochs.append(epochs_offset + iterations)

					# Compute validation cost
					costs = []
					for sequence, user_id in dataset.validation_set(epochs=1):
						top_k = self.top_k_recommendations(sequence[:len(sequence)//2], user_id=user_id)
						costs.append(int(sequence[len(sequence)//2][0] in top_k))
						# print('---------------')
						# print(top_k)
						# print(sequence[len(sequence)//2][0])
						
					last_cost = np.mean(costs)
					val_costs.append(last_cost)

					# Print info
					self._print_progress(iterations, epochs[-1], start_time, val_costs)

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

	def _print_progress(self, iterations, epochs, start_time, val_costs):
		'''Print learning progress in terminal
		'''
		print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
		print("Last val cost : ", val_costs[-1])
		print("Mean val cost : ", np.mean(val_costs))
		print('-----------------')

		# Print on stderr for easier recording of progress
		print(iterations, epochs, time() - start_time, "n/a", val_costs[-1], file=sys.stderr)

	def save(self, filename):
		'''Save the word2vec object into a file
		'''
		print('Save model in ' + filename)
		self.w2v_model.save(filename)

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
		self.w2v_model = Word2Vec.load(filename)