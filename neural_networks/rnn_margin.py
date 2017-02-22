from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import random
from time import time
import rnn_base as rnn
from sparse_lstm import *

class RNNMargin(rnn.RNNBase):
	'''

	OPTIONS
	-------
	balance: float, default 1, balance between the weight of false negative and false positive on the loss function.
		e.g. if balance = 1, both have the same weight, 
		if balance = 0, only false negative have an impact, 
		if balance = 2, false positive have twice as much weight as false negative.
	popularity_based: bool, default False, choose wether the target value of negatives depends on their popularity.
		if False, the target value of all negatives is 0 (versus 1 for the positives)
		if True, the target value of item i is min(1 - p_i, (1 - min_access) * p_i / min_access), where p_i = fraction of users who consumed that item.
	min_access: float in (0,1), default 0.05, parameter used in the formula for the target value of negatives in the popularity based case.
		Represent the minimum fraction of users that has access to any item.
		e.g. min_access=0.05 means that there is no item accessible by less than 5% of the users.
	n_targets: int or inf, default 1, number of items in the continuation of the sequence that will be used as positive target.

	'''
	
	def __init__(self, loss_function="hinge", balance=1., popularity_based=False, min_access=0.05, n_targets=1, **kwargs):
		super(RNNMargin, self).__init__(**kwargs)
		
		self.balance = balance
		self.popularity_based = popularity_based
		self.min_access = min_access
		self.n_targets = n_targets
		if loss_function is None:
			loss_function = "hinge"
		self.loss_function_name = loss_function
		if loss_function == "hinge":
			self.loss_function = self._hinge_loss
		elif loss_function == "logit":
			self.loss_function = self._logit_loss
		elif loss_function == "logsig":
			self.loss_function = self._logsigmoid_loss
		else:
			raise ValueError('Unknown loss function')

		self.name = "RNN multi-targets"

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "rnn_multitarget_"+self.loss_function_name+"_b"+str(self.balance)
		if self.popularity_based:
			filename += '_pb_ma'+str(self.min_access)
		return filename + "_" + self._common_filename(epochs)

	def _hinge_loss(self, predictions, targets, weights):
		return T.nnet.relu((predictions - targets) * weights).sum(axis=-1)

	def _logit_loss(self, predictions, targets, weights):
		return (T.nnet.sigmoid(predictions - targets) * weights).sum(axis=-1)

	def _logsigmoid_loss(self, predictions, targets, weights):
		return -T.log(T.nnet.sigmoid((targets - predictions) * weights)).sum(axis=-1)

	def _prepare_networks(self, n_items):
		''' Prepares the building blocks of the RNN, but does not compile them:
		self.l_in : input layer
		self.l_mask : mask of the input layer
		self.target : target of the network
		self.l_out : output of the network
		self.cost : cost function
		'''

		self.n_items = n_items
		
		# The input is composed of to parts : the on-hot encoding of the movie, and the features of the movie
		self.l_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_length, self._input_size()))
		# The input is completed by a mask to inform the LSTM of the length of the sequence
		self.l_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_length))

		# recurrent layer
		if not self.use_movies_features:
			l_recurrent = self.recurrent_layer(self.l_in, self.l_mask, true_input_size=self.n_items + self._n_optional_features(), only_return_final=True)
		else:
			l_recurrent = self.recurrent_layer(self.l_in, self.l_mask, true_input_size=None, only_return_final=True)

		# l_last_slice gets the last output of the recurrent layer
		l_last_slice = l_recurrent
		# l_last_slice = lasagne.layers.SliceLayer(l_recurrent, -1, 1)

		# Theano tensor for the targets
		target = T.fmatrix('multiple_target_output')
		target_weight = T.fmatrix('target_weight')
		self.exclude = T.fmatrix('excluded_items')
		self.theano_inputs = [self.l_in.input_var, self.l_mask.input_var, target, target_weight, self.exclude]
		
		# The sliced output is then passed through linear layer to obtain the right output size
		self.l_out = lasagne.layers.DenseLayer(l_last_slice, num_units=self.n_items, nonlinearity=None)
					
		# lasagne.layers.get_output produces a variable for the output of the net
		network_output = lasagne.layers.get_output(self.l_out)

		# loss function
		self.cost = self.loss_function(network_output, target, target_weight).mean()
		

	def _prepare_input(self, sequences):
		''' Sequences is a list of [user_id, input_sequence, targets]
		'''

		batch_size = len(sequences)

		# Shape return variables
		X = np.zeros((batch_size, self.max_length, self._input_size()), dtype=self._input_type) # input of the RNN
		mask = np.zeros((batch_size, self.max_length)) # mask of the input (to deal with sequences of different length)
		Y = np.zeros((batch_size, self.n_items), dtype=theano.config.floatX)
		weight = np.zeros((batch_size, self.n_items), dtype=theano.config.floatX)
		exclude = np.zeros((batch_size, self.n_items), dtype=theano.config.floatX)

		
		for i, sequence in enumerate(sequences):
			user_id, in_seq, target = sequence
			seq_features = np.array(map(lambda x: self._get_features(x, user_id), in_seq))
			X[i, :len(in_seq), :] = seq_features # Copy sequences into X
			mask[i, :len(in_seq)] = 1
			exclude[i, [j[0] for j in in_seq]] = 1

			# compute weight for false positive
			w = self.balance * len(target) / (self.n_items - len(target) - len(in_seq))

			weight[i,:] = w
			weight[i, [t[0] for t in target]] = -1
			if self.interactions_are_unique:
				weight[i, [t[0] for t in in_seq]] = 0
			

			Y[i, :] = self._default_target()
			Y[i, [t[0] for t in target]] = 1
			if self.interactions_are_unique:
				Y[i, [t[0] for t in in_seq]] = 0
			

		# weight *= 10e3
		return (X, mask.astype(theano.config.floatX), Y, weight, exclude)

	def _default_target(self):

		if not hasattr(self, '__default_target'):
			if not self.popularity_based:
				self.__default_target = np.zeros(self.n_items)
			else:
				num_users = self.dataset.training_set.n_users
				view_prob = self.dataset.item_popularity / num_users
				self.__default_target = np.minimum(1 - view_prob, (1 - self.min_access) * view_prob / self.min_access)

		return self.__default_target
