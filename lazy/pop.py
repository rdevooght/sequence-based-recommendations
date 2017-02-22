from __future__ import print_function
import collections
import numpy as np
import scipy.sparse as ssp
from copy import deepcopy
import os
from .lazy import Lazy
from .utils import top_k, get_sparse_vector


class Pop(Lazy):
	"""
	"""
	def __init__(self, **kwargs):
		super(Pop, self).__init__(**kwargs)
		self.name = "Pop"

	def _get_model_filename(self, *args):
		return "pop"

	def prepare_model(self, dataset):
		'''Load the data from the training file into a format adapted for the KNN methods.
		'''

		self._items_pop = np.zeros(dataset.n_items)
		for triplet in dataset.training_set_triplets():
			self._items_pop[triplet['item_id']] += 1

	def top_k_recommendations(self, sequence, k=10, exclude=None, **kwargs):
				
		if exclude is None:
			exclude = []

		items_pop = deepcopy(self._items_pop)

		items_pop[exclude] = -np.inf
		items_pop[[i[0] for i in sequence]] = -np.inf

		return list(np.argpartition(-items_pop, range(k))[:k])
