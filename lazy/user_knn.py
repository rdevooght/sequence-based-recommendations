from __future__ import print_function

import numpy as np
import scipy.sparse as ssp
import os.path
from .lazy import Lazy
from .utils import top_k, get_sparse_vector


class UserKNN(Lazy):
	"""
	"""
	def __init__(self, similarity_measure='cosine', neighborhood_size=80, **kwargs):
		super(UserKNN, self).__init__(**kwargs)
		
		self.similarity_measure = similarity_measure
		self.neighborhood_size = neighborhood_size		

		self.name = "UserKNN"

	def _get_model_filename(self, *args):
		return "UKNN_ns"+str(self.neighborhood_size)+"_"+self.similarity_measure

	def prepare_model(self, dataset):
		'''Load the data from the training file into a format adapted for the KNN methods.
		'''
		filename = dataset.dirname + 'data/train_set_triplets'
		if os.path.isfile(filename + '.npy'):
			file_content = np.load(filename + '.npy')
		else:
			file_content = np.loadtxt(filename)
			np.save(filename, file_content)

		#self.user_item = ssp.coo_matrix((file_content[:,2], (file_content[:,0], file_content[:,1]))).tocsr()
		self.binary_user_item = ssp.coo_matrix((np.ones(file_content.shape[0]), (file_content[:,0], file_content[:,1]))).tocsr()

		del file_content

		self.n_items = self.binary_user_item.shape[1]
		self.n_users = self.binary_user_item.shape[0]

	def _items_count_per_user(self):
		if not hasattr(self, '__items_count_per_user'):
			self.__items_count_per_user = np.asarray(self.binary_user_item.sum(axis=1)).ravel()
		return self.__items_count_per_user

	def similarity_with_users(self, sequence):
		'''Compute the similarity of each user with the sequence recieved in parameter
		'''
		sparse_sequence = get_sparse_vector([i[0] for i in sequence], self.n_items)
		overlap = self.binary_user_item.dot(sparse_sequence).toarray().ravel()
		overlap[overlap != 0] /= np.sqrt(self._items_count_per_user()[overlap != 0])
		return overlap

	def top_k_recommendations(self, sequence, k=10, exclude=None, **kwargs):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''
		if exclude is None:
			exclude = []

		sim_with_users = self.similarity_with_users(sequence)
		nearest_neighbors = top_k(sim_with_users, self.neighborhood_size)
		sim_with_users = get_sparse_vector(nearest_neighbors, self.n_users, values=sim_with_users[nearest_neighbors])
		sim_with_items = self.binary_user_item.T.dot(sim_with_users).toarray().ravel()

		sim_with_items[exclude] = -np.inf
		sim_with_items[[i[0] for i in sequence]] = -np.inf

		return list(np.argpartition(-sim_with_items, range(k))[:k])