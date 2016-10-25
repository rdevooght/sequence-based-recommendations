from __future__ import print_function

import numpy as np


class Lazy(object):
	"""Base for Lazy object.
	"""
	def __init__(self):
		super(Lazy, self).__init__()
		
		self.name = "Lazy base"

	def prepare_model(self, dataset):
		'''Must be called before using top_k_recommendations
		'''
		raise NotImplemented


	def load(self, *args, **kwargs):
		'''Nothing to do here
		'''
		return None

	def top_k_recommendations(self, sequence, k=10, **kwargs):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''
		raise NotImplemented