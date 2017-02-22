from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import re
import os
import glob
import sys
import random
from time import time
from sequence_noise import SequenceNoise
from update_manager import Adagrad
from recurrent_layers import RecurrentLayers
from target_selection import SelectTargets
from helpers import evaluation

#Lasagne Seed for Reproducibility
#lasagne.random.set_rng(np.random.RandomState(1))

# Maximum sequence Length
MAX_LENGTH = 200

# Feature added to the factorization features (user's rating, popularity, year, ...)
OTHER_FEATURES = None
MOVIES_FEATURES = None
USERS_FEATURES = None

# Number of sequence per batch
BATCH_SIZE = 10

def threaded_generator(generator, num_cached=50):
	import Queue
	queue = Queue.Queue(maxsize=num_cached)
	sentinel = object()  # guaranteed unique reference

	# define producer (putting items into queue)
	def producer():
		for item in generator:
			queue.put(item)
		queue.put(sentinel)

	# start producer (in a background thread)
	import threading
	thread = threading.Thread(target=producer)
	thread.daemon = True
	thread.start()

	# run as consumer (read items from queue, in current thread)
	item = queue.get()
	while item is not sentinel:
		yield item
		queue.task_done()
		item = queue.get()

class RNNBase(object):
	"""Base for RNN object.
	"""
	def __init__(self, 
		sequence_noise=SequenceNoise(),
		recurrent_layer=RecurrentLayers(),
		updater=Adagrad(),
		target_selection=SelectTargets(),
		interactions_are_unique=True,
		other_features=OTHER_FEATURES, 
		use_ratings_features=True, 
		movies_features=MOVIES_FEATURES, 
		use_movies_features=True, 
		users_features=USERS_FEATURES, 
		use_users_features=True, 
		max_length=MAX_LENGTH, 
		batch_size=BATCH_SIZE):
		super(RNNBase, self).__init__()
		
		self.other_features = other_features
		self.use_ratings_features = use_ratings_features
		self.movies_features = movies_features
		self.use_movies_features = use_movies_features
		self.users_features = users_features
		self.use_users_features = use_users_features
		self.max_length = max_length
		self.batch_size = batch_size
		self.sequence_noise = sequence_noise
		self.recurrent_layer = recurrent_layer
		self.updater = updater
		self.target_selection = target_selection
		self.interactions_are_unique = interactions_are_unique

		if self.use_movies_features:
			self._input_type = theano.config.floatX
		else:
			self._input_type = 'int32'

		self.name = "RNN base"

		self.metrics = {'recall': {'direction': 1},
			'sps': {'direction': 1},
			'user_coverage' : {'direction': 1},
			'item_coverage' : {'direction': 1},
			'ndcg' : {'direction': 1},
			'blockbuster_share' : {'direction': -1}
		}

	def prepare_model(self, dataset):
		'''Must be called before using train, load or top_k_recommendations
		'''
		self._prepare_networks(dataset.n_items)

	def _common_filename(self, epochs):
		'''Common parts of the filename accros sub classes.
		'''
		filename = "ml"+str(self.max_length)+"_bs"+str(self.batch_size)+"_ne"+str(epochs)+"_"+self.recurrent_layer.name + "_" + self.updater.name + "_" + self.target_selection.name
		
		if self.sequence_noise.name != "":
			filename += "_" + self.sequence_noise.name
		
		if not self.interactions_are_unique:
			filename += "_ri"

		if not (self.use_ratings_features or self.use_movies_features or self.use_users_features):
			filename += "_nf"
		if self.use_ratings_features:
			filename += "_rf"
		if self.use_movies_features:
			filename += "_mf"
		if self.use_users_features:
			filename += "_uf"
		return filename

	def top_k_recommendations(self, sequence, user_id=None, k=10, exclude=None):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		if exclude is None:
			exclude = []

		# Compile network if needed
		if not hasattr(self, 'predict_function'):
			self._compile_predict_function()

		# Prepare RNN input
		max_length_seq = sequence[-min(self.max_length, len(sequence)):]
		X = np.zeros((1, self.max_length, self._input_size()), dtype=self._input_type) # input of the RNN
		X[0, :len(max_length_seq), :] = np.array(map(lambda x: self._get_features(x, user_id), max_length_seq))
		mask = np.zeros((1, self.max_length)) # mask of the input (to deal with sequences of different length)
		mask[0, :len(max_length_seq)] = 1

		# Run RNN
		output = self.predict_function(X, mask.astype(theano.config.floatX))[0]

		# Put low similarity to viewed items to exclude them from recommendations
		if self.interactions_are_unique:
			output[[i[0] for i in sequence]] = -np.inf
		output[exclude] = -np.inf

		# find top k according to output
		return list(np.argpartition(-output, range(k))[:k])
	
	def set_dataset(self, dataset):
		self.dataset = dataset
		self.target_selection.set_dataset(dataset)

	def get_pareto_front(self, metrics, metrics_names):
		costs = np.zeros((len(metrics[metrics_names[0]]), len(metrics_names)))
		for i, m in enumerate(metrics_names):
			costs[:, i] = np.array(metrics[m]) * self.metrics[m]['direction']
		is_efficient = np.ones(costs.shape[0], dtype = bool)
		for i, c in enumerate(costs):
			if is_efficient[i]:
				is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)
		return np.where(is_efficient)[0].tolist()

	def _compile_train_function(self):
		''' Compile self.train. 
		self.train recieves a sequence and a target for every steps of the sequence, 
		compute error on every steps, update parameter and return global cost (i.e. the error).
		'''
		print("Compiling train...")
		# Compute AdaGrad updates for training
		all_params = lasagne.layers.get_all_params(self.l_out, trainable=True)
		updates = self.updater(self.cost, all_params)
		# Compile network
		self.train_function = theano.function(self.theano_inputs, self.cost, updates=updates, allow_input_downcast=True, name="Train_function", on_unused_input='ignore')
		print("Compilation done.")

	def _compile_predict_function(self):
		''' Compile self.predict, the deterministic rnn that output the prediction at the end of the sequence
		'''
		print("Compiling predict...")
		deterministic_output = lasagne.layers.get_output(self.l_out, deterministic=True)
		self.predict_function = theano.function([self.l_in.input_var, self.l_mask.input_var], deterministic_output, allow_input_downcast=True, name="Predict_function")
		print("Compilation done.")

	def _compile_test_function(self):
		''' Compile self.test_function, the deterministic rnn that output the k best scoring items
		'''
		print("Compiling test...")
		deterministic_output = lasagne.layers.get_output(self.l_out, deterministic=True)
		if self.interactions_are_unique:
			deterministic_output *= (1 - self.exclude)
		theano_test_function = theano.function(self.theano_inputs, deterministic_output, allow_input_downcast=True, name="Test_function", on_unused_input='ignore')
		
		def precision_test_function(theano_inputs, k=10):
			output = theano_test_function(*theano_inputs)
			ids = np.argpartition(-output, range(k), axis=-1)[0, :k]
			
			return ids

		self.test_function = precision_test_function

		print("Compilation done.")

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
		early_stopping=None,
		validation_metrics=['sps']):
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
 
		self.set_dataset(dataset)
		
		if len(set(validation_metrics) & set(self.metrics.keys())) < len(validation_metrics):
			raise ValueError('Incorrect validation metrics. Metrics must be chosen among: ' + ', '.join(self.metrics.keys()))

		# Compile network if needed
		if not hasattr(self, 'train_function'):
			self._compile_train_function()
		if not hasattr(self, 'test_function'):
			self._compile_test_function()
 
		# Load last model if needed
		iterations = 0
		epochs_offset = 0
		if load_last_model:
			epochs_offset = self.load_last(save_dir)
		 
		# Make batch generator
		#batch_generator = threaded_generator(self._gen_mini_batch(self.sequence_noise(dataset.training_set())))
		batch_generator = self._gen_mini_batch(self.sequence_noise(dataset.training_set()))
 
		start_time = time()
		next_save = int(progress)
		train_costs = []
		current_train_cost = []
		epochs = []
		metrics = {name:[] for name in self.metrics.keys()}
		filename = {}
 
		try: 
			while (time() - start_time < max_time and iterations < max_iter):
 
				# Train with a new batch
				try:
					batch = next(batch_generator)
					cost = self.train_function(*batch)
					if np.isnan(cost):
						raise ValueError("Cost is NaN")
					 
				except StopIteration:
					break
 
				current_train_cost.append(cost)
 
				# Check if it is time to save the model
				iterations += 1
 
				if time_based_progress:
					progress_indicator = int(time() - start_time)
				else:
					progress_indicator = iterations
 
				if progress_indicator >= next_save:
 
					if progress_indicator >= min_iterations:
 
						# Save current epoch
						epochs.append(epochs_offset + dataset.training_set.epochs)
 
						# Average train cost
						train_costs.append(np.mean(current_train_cost))
						current_train_cost = []
 
						# Compute validation cost
						metrics = self._compute_validation_metrics(metrics)
							
						# Print info
						self._print_progress(iterations, epochs[-1], start_time, train_costs, metrics, validation_metrics)

						# Save model
						run_nb = len(metrics[self.metrics.keys()[0]])-1
						if autosave == 'All':
							filename[run_nb] = save_dir + self._get_model_filename(round(epochs[-1], 3))
							self.save(filename[run_nb])
						elif autosave == 'Best':
							pareto_runs = self.get_pareto_front(metrics, validation_metrics)
							if run_nb in pareto_runs:
								filename[run_nb] = save_dir + self._get_model_filename(round(epochs[-1], 3))
								self.save(filename[run_nb])
								to_delete = [r for r in filename if r not in pareto_runs]
								for run in to_delete:
									try:
										os.remove(filename[run])
									except OSError:
										print('Warning : Previous model could not be deleted')
									del filename[run]

						if early_stopping is not None:
							# Stop if early stopping is triggered for all the validation metrics
							if all([early_stopping(epochs, metrics[m]) for m in validation_metrics]):
								break 
 
					# Compute next checkpoint
					if isinstance(progress, int):
						next_save += min(progress, max_progress_interval)
					else:
						next_save += min(max_progress_interval, next_save * (progress - 1))
		except KeyboardInterrupt:
			print('Training interrupted')
		 
		best_run = np.argmax(np.array(metrics[validation_metrics[0]]) * self.metrics[validation_metrics[0]]['direction'])
		return ({m: metrics[m][best_run] for m in self.metrics.keys()}, time()-start_time, filename[best_run])

	def _compute_validation_metrics(self, metrics):
		ev = evaluation.Evaluator(self.dataset, k=10)
		for batch_input, goal in self._gen_mini_batch(self.dataset.validation_set(epochs=1), test=True):
			predictions = self.test_function(batch_input)
			ev.add_instance(goal, predictions)

		metrics['recall'].append(ev.average_recall())
		metrics['sps'].append(ev.sps())
		metrics['ndcg'].append(ev.average_ndcg())
		metrics['user_coverage'].append(ev.user_coverage())
		metrics['item_coverage'].append(ev.item_coverage())
		metrics['blockbuster_share'].append(ev.blockbuster_share())

		return metrics

	def _gen_mini_batch(self, sequence_generator, test=False, max_reuse_sequence=np.inf):
		''' Takes a sequence generator and produce a mini batch generator.
		The mini batch have a size defined by self.batch_size, and have format of the input layer of the rnn.

		test determines how the sequence is splitted between training and testing
			test == False, the sequence is split randomly
			test == True, the sequence is split in the middle

		if test == False, max_reuse_sequence determines how many time a single sequence is used in the same batch.
			with max_reuse_sequence = inf, one sequence will be used to make the whole batch (if the sequence is long enough)
			with max_reuse_sequence = 1, each sequence is used only once in the batch
		N.B. if test == True, max_reuse_sequence = 1 is used anyway

		
		'''

		while True:

			j = 0
			sequences = []
			batch_size = self.batch_size
			if test:
				batch_size = 1
			while j < batch_size:

				sequence, user_id = next(sequence_generator)

				# finds the lengths of the different subsequences
				if not test:
					seq_lengths = sorted(random.sample(xrange(2, len(sequence)), min([batch_size - j, len(sequence) - 2, max_reuse_sequence])))
				else:
					seq_lengths = [int(len(sequence) / 2)] 

				skipped_seq = 0
				for l in seq_lengths:
					target = self.target_selection(sequence[l:], test=test)
					if len(target) == 0:
						skipped_seq += 1
						continue
					start = max(0, l - self.max_length) # sequences cannot be longer than self.max_lenght
					sequences.append([user_id, sequence[start:l], target])

				j += len(seq_lengths) - skipped_seq

			if test:
				yield self._prepare_input(sequences), [i[0] for i in sequence[seq_lengths[0]:]]
			else:
				yield self._prepare_input(sequences)

	def _print_progress(self, iterations, epochs, start_time, train_costs, metrics, validation_metrics):
		'''Print learning progress in terminal
		'''
		print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
		print("Last train cost : ", train_costs[-1])
		for m in self.metrics:
			print(m, ': ', metrics[m][-1])
			if m in validation_metrics:
				print('Best ', m, ': ', max(np.array(metrics[m])*self.metrics[m]['direction'])*self.metrics[m]['direction'])
		print('-----------------')

		# Print on stderr for easier recording of progress
		print(iterations, epochs, time() - start_time, train_costs[-1], ' '.join(map(str, [metrics[m][-1] for m in self.metrics])), file=sys.stderr)


	def _get_model_filename(self, iterations):
		'''Return the name of the file to save the current model
		'''
		raise NotImplemented

	def _prepare_networks(self):
		''' Prepares the building blocks of the RNN, but does not compile them:
		self.l_in : input layer
		self.l_mask : mask of the input layer
		self.target : target of the network
		self.l_out : last output of the network
		self.cost : cost function

		and maybe others
		'''
		raise NotImplemented
	   
		
	def _compile_train_network(self):
		''' Compile self.train. 
		self.train recieves a sequence and a target for every steps of the sequence, 
		compute error on every steps, update parameter and return global cost (i.e. the error).
		'''
		raise NotImplemented
		

	def _compile_predict_network(self):
		''' Compile self.predict, the deterministic rnn that output the prediction at the end of the sequence
		'''
		raise NotImplemented
		


	def save(self, filename):
		'''Save the parameters of a network into a file
		'''
		print('Save model in ' + filename)
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))
		param = lasagne.layers.get_all_param_values(self.l_out)
		f = file(filename, 'wb')
		cPickle.dump(param,f,protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

	def load_last(self, save_dir):
		'''Load last model from dir
		'''
		def extract_number_of_batches(filename):
			m = re.search('_nb([0-9]+)_', filename)
			return int(m.group(1))

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
		f = file(filename, 'rb')
		param = cPickle.load(f)
		f.close()
		lasagne.layers.set_all_param_values(self.l_out, [i.astype(theano.config.floatX) for i in param])
	
	def _n_items_features(self):
		'''Number of movies features
		'''
		if not hasattr(self, '__n_items_features'):
			f = self._get_movies_features((0,1)) # get features for movie 0, rating 1
			self.__n_items_features = len(f)
		return self.__n_items_features

	def _n_ratings_features(self):
		'''Number of features linked to rating
		'''
		if not hasattr(self, '__n_other_features'):
			f = self._get_ratings_features((0,1)) # get features for movie 0, rating 1
			self.__n_ratings_features = len(f)
		return self.__n_ratings_features

	def _n_users_features(self):
		'''Number of users features
		'''
		if not hasattr(self, '__n_users_features'):
			f = self._get_user_features(0) # get features for user 0
			self.__n_users_features = len(f)
		return self.__n_users_features

	def _n_optional_features(self):
		''' Number of optional features
		'''
		return self._n_ratings_features() + self._n_users_features() + self._n_items_features()
	
	def _get_movies_features(self, item):
		'''Get the "movies features" of an item, i.e. [year, genre]
		The year is a one-hot-encoding with 8 neurons: before the 50s, the 50s, the 60s, ..., the 2000s, and the 2010s
		'''
		
		def int2list(val, length):
			f = np.zeros(length)
			f[val - 1] = 1
			return f
		
		def year_to_decade(year):
			decade = np.zeros(8)
			if year < 1950:
				decade[0] = 1
			elif year < 2000:
				i = int((year - 1900) / 10) - 4
				decade[i] = 1
			else:
				i = int((year - 2000) / 10) + 6
				decade[i] = 1
			return decade

		if not self.use_movies_features:
			return []
		else :
			item_id, rating = item
			decade = year_to_decade(self.movies_features[item_id, 1])
			genre = self.movies_features[item_id, 2:]
			avg_rating = int2list(round(self.other_features[item_id, 1]*2), 10)
			popularity = int2list(self.other_features[item_id, 3], 10)
			return np.concatenate((decade, genre, avg_rating, popularity))

	def _get_ratings_features(self, item):
		'''Get the "other features" of an item, i.e. [personal_rating on a scale of ten, average_rating on a scale of ten, popularity on a log scale of ten]
		'''
		
		def int2list(val, length):
			f = np.zeros(length)
			f[int(val) - 1] = 1
			return f

		item_id, rating = item

		if not self.use_ratings_features:
			return []
		else :
			rating = int2list(round(rating*2), 10)
			return rating

	def _get_user_features(self, user_id):
		'''Get the features of a user [sex, age, occupation]
		'''
		
		def int2list(val, length):
			f = np.zeros(length)
			f[val] = 1
			return f

		if not self.use_users_features:
			return []
		else :
			sex = int2list(self.users_features[user_id, 1], 2)
			age = int2list(self.users_features[user_id, 2], 7)
			occupation = int2list(self.users_features[user_id, 3], 21)
			return np.concatenate((sex, age, occupation))

	def _get_optional_features(self, item, user_id):
		return np.concatenate((self._get_ratings_features(item), self._get_movies_features(item), self._get_user_features(user_id)))

	def _input_size(self):
		''' Returns the number of input neurons
		'''
		
		if self.use_movies_features:
			return self.n_items + self._n_optional_features()
		else:
			return len(np.nonzero(self._get_optional_features((0, 1), 0))[0]) + 1

	def _get_features(self, item, user_id):
		'''Change a tuple (item_id, rating) into a list of features to feed into the RNN
		features have the following structure: [one_hot_encoding, personal_rating on a scale of ten, average_rating on a scale of ten, popularity on a log scale of ten]
		'''

		item_id, rating = item

		if self.use_movies_features:
			one_hot_encoding = np.zeros(self.n_items)
			one_hot_encoding[item_id] = 1
			
			return np.concatenate((one_hot_encoding, self._get_optional_features(item, user_id)))
		else:
			one_hot_encoding = [item_id]

			optional_features = self._get_optional_features(item, user_id)
			optional_features_ids = np.nonzero(optional_features)[0]
			
			return np.concatenate((one_hot_encoding, optional_features_ids + self.n_items))
