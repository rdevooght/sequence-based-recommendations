from __future__ import division
import numpy as np
import theano
import theano.tensor as T
import random
import os.path

# Data directory
DEFAULT_DIR = '../../data/'


class DataHandler(object):
	''' Prepare data for the differents algorithms.
	Give easy access to training, validation and test set and to information about the dataset 
	such as number of users, items and interactions.
	'''

	def __init__(self, dirname, extended_training_set=False, shuffle_training=False):
		'''

		Parameter
		---------

		dirname: str
			Directory where the dataset can be found.
			If dirname does not correspond to an existing path, DEFAULT_DIR+dirname will be looked for.
			If both dirname and DEFAULT_DIR+dirname are existing path, a warning will be printed.
			The directory should contains at least the following sub folders:
			- data/ where the dataset files can be found
			- models/ where the models are stored during training
			- results/ where the results are stored during testing

		extended_training_set: boolean
			If True, the extended training set is used, i.e. the file "train_set_sequences+" is loaded instead of "train_set_sequences".

		shuffle_training: boolean
			If True, the order of the training sequences is shuffled between each pass.
		'''
		super(DataHandler, self).__init__()

		self.dirname = self._get_path(dirname)

		self.extended_training_set = extended_training_set
		if extended_training_set:
			self.training_set = SequenceGenerator(self.dirname+'data/train_set_sequences+', shuffle=shuffle_training)
		else:
			self.training_set = SequenceGenerator(self.dirname+'data/train_set_sequences', shuffle=shuffle_training)
		self.validation_set = SequenceGenerator(self.dirname+'data/val_set_sequences')
		self.test_set = SequenceGenerator(self.dirname+'data/test_set_sequences')

		self._load_stats()

	def training_set_triplets(self):
		with open(self.dirname + 'data/train_set_triplets') as f:
			for line in f:
				line = line.split()
				yield {'user_id': int(line[0]), 'item_id': int(line[1]), 'rating': float(line[2])}

	@property
	def item_popularity(self):
		'''Returns the number of occurences of an item in the training set.
		'''

		if not hasattr(self.training_set, '_item_pop'):
			if os.path.isfile(self.dirname + 'data/training_set_item_popularity.npy'):
				self.training_set._item_pop = np.load(self.dirname + 'data/training_set_item_popularity.npy')
			else:
				self.training_set._item_pop = np.zeros(self.n_items)
				with open(self.dirname + 'data/train_set_triplets') as f:
					for line in f:
						self.training_set._item_pop[int(line.split()[1])] += 1
				np.save(self.dirname + 'data/training_set_item_popularity.npy', self.training_set._item_pop)

		return self.training_set._item_pop

	def _get_path(self, dirname):
		''' Choose between dirname and DEFAULT_DIR+dirname.
		'''
		if os.path.exists(dirname) and not os.path.exists(DEFAULT_DIR+dirname+'/'):
			return dirname
		if not os.path.exists(dirname) and os.path.exists(DEFAULT_DIR+dirname+'/'):
			return DEFAULT_DIR+dirname+'/'
		if os.path.exists(dirname) and os.path.exists(DEFAULT_DIR+dirname+'/'):
			print('WARNING: ambiguous directory name, both "'+dirname+'" and "'+DEFAULT_DIR+dirname+'" exist. "'+dirname+'" is used.')
			return dirname
		
		raise ValueError('Dataset not found')

	def _load_stats(self):
		''' Load informations about the dataset from dirname/data/stats
		'''
		with open(self.dirname+'data/stats', 'r') as f:
			_ = f.readline() # Line with column titles
			self.n_users, self.n_items, self.n_interactions, self.longest_sequence = map(int, f.readline().split()[1:])
			self.training_set.n_users, self.training_set.n_items, self.training_set.n_interactions, self.training_set.longest_sequence = map(int, f.readline().split()[1:])
			self.validation_set.n_users, self.validation_set.n_items, self.validation_set.n_interactions, self.validation_set.longest_sequence = map(int, f.readline().split()[1:])
			self.test_set.n_users, self.test_set.n_items, self.test_set.n_interactions, self.test_set.longest_sequence = map(int, f.readline().split()[1:])

			if self.extended_training_set:
				#Those values are unfortunately inexact
				self.training_set.n_users, self.training_set.n_items = self.n_users, self.n_items
				self.training_set.n_interactions += (self.validation_set.n_interactions + self.test_set.n_interactions)//2

class SequenceGenerator(object):
	"""docstring for SequenceGenerator"""
	def __init__(self, filename, shuffle=False):
		super(SequenceGenerator, self).__init__()
		self.filename = filename
		self.shuffle = shuffle
		self.epochs = 0.

	def load(self):

		self.lines = []
		# self.max_length = 0
		# self.max_user_id = 0
		# self.max_item_id = 0

		with open(self.filename, 'r') as f:
			for sequence in f:
				self.lines.append(sequence)
				# self.max_length = max(self.max_length, (len(sequence.split()) - 1) / 2)
				# self.max_user_id = max(self.max_user_id, int(sequence.split()[0]))
				# self.max_item_id = max(self.max_item_id, max(map(int, sequence.split()[1::2])))
	
	def __call__(self, min_length=2, max_length=None, length_choice='max', subsequence='contiguous', epochs=np.inf):
		if not hasattr(self, 'lines'):
			self.load()

		counter = 0
		self.epochs = 0.
		while counter < epochs:
			counter += 1
			print("Opening file ({})".format(counter))
			if self.shuffle:
				random.shuffle(self.lines)
			for j, sequence in enumerate(self.lines):
				
				self.epochs = counter - 1 + j / len(self.lines)
				
				# Express sequence as a list of tuples (movie_id, rating)
				sequence = sequence.split()
				user_id = sequence[0]
				sequence = sequence[1:]
				sequence = [[int(sequence[2*i]), float(sequence[2*i + 1])] for i in range(int(len(sequence) / 2))]

				# Determine length of the sequence to be returned
				if max_length == None:
					this_max_length = len(sequence)
				else:
					this_max_length = max_length

				if len(sequence) < min_length:
					continue
				if (length_choice == 'random'):
					length = np.random.randint(min_length, min(this_max_length, len(sequence)) + 1)
				elif (length_choice == 'max'):
					length = min(this_max_length, len(sequence))
				else:
					raise ValueError('Unrecognised length_choice option. Authorised values are "random" and "max" ')
				
				# Extract subsequence if needed
				if (length < len(sequence)):
					if subsequence == 'random':
						sequence = [ sequence[i] for i in sorted(random.sample(xrange(len(sequence)), length)) ]
					elif subsequence == 'contiguous':
						start = np.random.randint(0, len(sequence) - length + 1)
						sequence = sequence[start:start+length]
					elif subsequence == 'begining':
						sequence = sequence[:length]
					else:
						raise ValueError('Unrecognised subsequence option. Authorised values are "random", "contiguous" and "begining".')

				yield sequence, user_id
		