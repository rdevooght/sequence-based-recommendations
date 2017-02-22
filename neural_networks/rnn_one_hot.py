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

class RNNOneHot(rnn.RNNBase):
	"""RNNOneHot are recurrent neural networks that do not depend on the factorization: they are based on one-hot encoding.

	The parameters specific to the RNNOneHot are:
		diversity_bias: a float in [0, inf) that tunes how the cost function of the network is biased towards less seen movies.
			In practice, the classification error given by the categorical cross-entropy is divided by exp(diversity_bias * popularity (on a scale from 1 to 10)).
			This will reduce the error associated to movies with a lot of views, putting therefore more importance on the ability of the network to correctly predict the rare movies.
			A diversity_bias of 0 produces the normal behavior, with no bias.
	"""
	def __init__(self, diversity_bias=0.0, regularization=0.0, **kwargs):
		super(RNNOneHot, self).__init__(**kwargs)
		
		self.diversity_bias = np.cast[theano.config.floatX](diversity_bias)
		
		self.regularization = regularization

		self.name = "RNN with categorical cross entropy"

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "rnn_cce_db"+str(self.diversity_bias)+"_r"+str(self.regularization)+"_"+self._common_filename(epochs)
		return filename

	def _prepare_networks(self, n_items):
		''' Prepares the building blocks of the RNN, but does not compile them:
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
		target = T.ivector('target_output')
		target_popularity = T.fvector('target_popularity')
		self.exclude = T.fmatrix('excluded_items')
		self.theano_inputs = [self.l_in.input_var, self.l_mask.input_var, target, target_popularity, self.exclude]
		
		
		# The sliced output is then passed through linear layer to obtain the right output size
		self.l_out = lasagne.layers.DenseLayer(l_last_slice, num_units=self.n_items, nonlinearity=lasagne.nonlinearities.softmax)
					
		# lasagne.layers.get_output produces a variable for the output of the net
		network_output = lasagne.layers.get_output(self.l_out)

		# loss function
		self.cost = (T.nnet.categorical_crossentropy(network_output, target) / target_popularity).mean()

		if self.regularization > 0.:
			self.cost += self.regularization * lasagne.regularization.l2(self.l_out.b)
			# self.cost += self.regularization * lasagne.regularization.regularize_layer_params(self.l_out, lasagne.regularization.l2)
		elif self.regularization < 0.:
			self.cost -= self.regularization * lasagne.regularization.l1(self.l_out.b)
			# self.cost -= self.regularization * lasagne.regularization.regularize_layer_params(self.l_out, lasagne.regularization.l1)
		

	

	def _prepare_input(self, sequences):
		''' Sequences is a list of [user_id, input_sequence, targets]
		'''

		batch_size = len(sequences)

		# Shape return variables
		X = np.zeros((batch_size, self.max_length, self._input_size()), dtype=self._input_type) # input of the RNN
		mask = np.zeros((batch_size, self.max_length)) # mask of the input (to deal with sequences of different length)
		Y = np.zeros((batch_size,), dtype='int32') # output target
		pop = np.zeros((batch_size,)) # output target
		exclude = np.zeros((batch_size, self.n_items), dtype=theano.config.floatX)

		
		for i, sequence in enumerate(sequences):
			user_id, in_seq, target = sequence
			seq_features = np.array(map(lambda x: self._get_features(x, user_id), in_seq))
			X[i, :len(in_seq), :] = seq_features # Copy sequences into X
			mask[i, :len(in_seq)] = 1
			Y[i] = target[0][0] # id of the first and only target
			pop[i] = self.dataset.item_popularity[target[0][0]] ** self.diversity_bias
			exclude[i, [j[0] for j in in_seq]] = 1

		return (X, mask.astype(theano.config.floatX), Y, pop.astype(theano.config.floatX), exclude)
