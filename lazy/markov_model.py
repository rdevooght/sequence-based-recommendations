from __future__ import print_function
import collections
import numpy as np
import scipy.sparse as ssp
from copy import deepcopy
from .lazy import Lazy
from .utils import top_k, get_sparse_vector


class MarkovModel(Lazy):
	"""
	"""
	def __init__(self, **kwargs):
		super(MarkovModel, self).__init__(**kwargs)
		
		self.previous_recommendations = dict()	

		self.name = "MarkovModel"

	def _get_model_filename(self, *args):
		return "MM"
	
	def prepare_model(self, dataset):
		'''Load the data from the training file into a format adapted for the MM predictions.
		'''
		self.n_items = dataset.n_items

		self.sequences = []

		with open(dataset.training_set.filename, 'r') as f:
			for sequence in f:
				sequence = sequence.split()
				items = map(int, sequence[1::2])
				s = dict()
				for i in range(len(items)-1):
					s[items[i]] = items[i+1]
				self.sequences.append(s)

	def get_all_recommendations(self, item):
		all_recommendations = []
		for s in self.sequences:
			if item in s:
				all_recommendations.append(s[item])
		all_recommendations = collections.Counter(all_recommendations)
		del all_recommendations[None]
		self.previous_recommendations[item] = all_recommendations


	def top_k_recommendations(self, sequence, k=10, exclude=None, **kwargs):
		if exclude is None:
			exclude = []
		
		last_item = int(sequence[-1][0])
		if last_item not in self.previous_recommendations:
			self.get_all_recommendations(last_item)
		
		all_recommendations = deepcopy(self.previous_recommendations[last_item])
		for s in sequence:
			all_recommendations[int(s[0])] = 0
		for i in exclude:
			all_recommendations[i] = 0

		ranking = np.zeros(self.n_items)
		for i, x in enumerate(all_recommendations.most_common(k)):
			ranking[x[0]] = k-i
		return np.argpartition(-ranking, range(k))[:k]
