from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import random
from bisect import bisect
from time import time
import rnn_base as rnn
from sparse_lstm import *

class RNNSampling(rnn.RNNBase):
	"""RNNSampling have a loss function that uses a sampling procedure.
	BPR or TOP1
	"""
	def __init__(self, loss_function="Blackout", sampling=32, last_layer_tanh=False, last_layer_init=1., diversity_bias=0.0, sampling_bias=0., **kwargs):
		'''
		Parameters
		----------
		loss_function: "BPR" or "TOP1" or "Blackout"
			Choice between 3 loss functions:
			- BPR, as used in "Session-based Recommendations with Recurrent Neural Networks", Hidasi, B. et al., 2016
			- TOP1, defined in "Session-based Recommendations with Recurrent Neural Networks", Hidasi, B. et al., 2016
			- Blackout, discriminative loss function defined in "BlackOut: Speeding up Recurrent Neural Network Language Models With Very Large Vocabularies", Ji, S. et al., 2015 (equation 6)
		
		sampling: integer > 0 or float in (0,1)
			Number of items to sample in the computation of the loss function.
			If sampling is a float in (0,1), it is interpreted as the fraction of items to use.
		sampling_bias: float
			Items are sampled with a probability proportional to their frequency to the power of the sampling_bias.

		'''
		super(RNNSampling, self).__init__(**kwargs)
		
		self.last_layer_init = last_layer_init
		self.last_layer_tanh =last_layer_tanh
		self.diversity_bias = diversity_bias
		self.sampling = sampling
		self.sampling_bias = sampling_bias
		if loss_function is None:
			loss_function = "Blackout"
		self.loss_function_name = loss_function
		if loss_function == "BPR":
			self.loss_function = self._BPR_loss
		elif loss_function == "BPRI":
			self.loss_function = self._BPRI_loss
		elif loss_function == "TOP1":
			self.loss_function = self._TOP1_loss
		elif loss_function == "Blackout":
			self.loss_function = self._blackout_loss
		else:
			raise ValueError("Unknown loss function")

		
		self.name = "RNN with sampling loss"

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "rnn_sampling_"+self.loss_function_name+"_"
		if self.sampling_bias > 0.:
			filename += "p" + str(self.sampling_bias)
		filename += "s"+str(self.sampling)+"_ini"+str(self.last_layer_init)+"_db"+str(self.diversity_bias)
		return filename + "_" + self._common_filename(epochs)

	def _blackout_loss(self, predictions, targets):
		predictions = T.nnet.softmax(predictions)
		pos = T.nnet.categorical_crossentropy(predictions, targets)
		neg = T.log(1 - predictions)
		return pos - neg[:, targets.shape[0]:].sum(axis=-1)

	def _BPRI_loss(self, predictions, targets):
		if self.last_layer_tanh:
			predictions = T.tanh(predictions)
		diff = (predictions - T.diag(predictions).dimshuffle([0,'x']))[:, targets.shape[0]:]
		return (T.log(T.nnet.sigmoid(diff))).mean(axis=-1)

	def _BPR_loss(self, predictions, targets):
		if self.last_layer_tanh:
			predictions = T.tanh(predictions)
		diff = (predictions - T.diag(predictions).dimshuffle([0,'x']))[:, targets.shape[0]:]
		return -(T.log(T.nnet.sigmoid(-diff))).mean(axis=-1)

	def _TOP1_loss(self, predictions, targets):
		if self.last_layer_tanh:
			predictions = T.tanh(predictions)
		diff = (predictions - T.diag(predictions).dimshuffle([0,'x']))[:, targets.shape[0]:]
		reg = T.sqr(predictions[:, targets.shape[0]:])
		return (T.nnet.sigmoid(diff) + T.nnet.sigmoid(reg)).mean(axis=-1)

	def _prepare_networks(self, n_items):
		''' Prepares the building blocks of the RNN, but does not compile them:
		self.l_in : input layer
		self.l_mask : mask of the input layer
		self.target : target of the network
		self.l_out : output of the network
		self.cost : cost function
		'''

		self.n_items = n_items
		if self.sampling < 1:
			self.effective_sampling = int(self.sampling * self.n_items)
		else:
			self.effective_sampling = int(self.sampling)
		
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
		samples = T.ivector('samples')
		self.exclude = T.fmatrix('excluded_items')
		target_popularity = T.fvector('target_popularity')
		self.theano_inputs = [self.l_in.input_var, self.l_mask.input_var, target, samples, target_popularity, self.exclude]
		
		# The sliced output is then passed through linear layer to obtain the right output size
		self.l_out = BlackoutLayer(l_last_slice, num_units=self.n_items, num_outputs=self.sampling, nonlinearity=None, W=lasagne.init.GlorotUniform(gain=self.last_layer_init))

		# lasagne.layers.get_output produces a variable for the output of the net
		network_output = lasagne.layers.get_output(self.l_out, targets = target, samples=samples)

		# loss function
		self.cost = (self.loss_function(network_output,np.arange(self.batch_size)) / target_popularity).mean()
		

	def _compile_test_function(self):
		''' Differs from base test function because of the added softmax operation
		'''
		print("Compiling test...")
		deterministic_output = T.nnet.softmax(lasagne.layers.get_output(self.l_out, deterministic=True))
		if self.interactions_are_unique:
			deterministic_output *= (1 - self.exclude)

		theano_test_function = theano.function(self.theano_inputs, deterministic_output, allow_input_downcast=True, name="Test_function", on_unused_input='ignore')
		
		def precision_test_function(theano_inputs, k=10):
			output = theano_test_function(*theano_inputs)
			ids = np.argpartition(-output, range(k), axis=-1)[0, :k]
			
			return ids

		self.test_function = precision_test_function
		print("Compilation done.")

	def _popularity_sample(self):
		if not hasattr(self, '_cumsum'):
			self._cumsum = np.cumsum(np.power(self.dataset.item_popularity, self.sampling_bias))

		return bisect(self._cumsum, random.uniform(0, self._cumsum[-1]))

	def _prepare_input(self, sequences):
		''' Sequences is a list of [user_id, input_sequence, targets]
		'''

		batch_size = len(sequences)

		# Shape return variables
		X = np.zeros((batch_size, self.max_length, self._input_size()), dtype=self._input_type) # input of the RNN
		mask = np.zeros((batch_size, self.max_length)) # mask of the input (to deal with sequences of different length)
		Y = np.zeros((batch_size,), dtype='int32') # output target
		pop = np.zeros((batch_size,)) # output target popularity
		exclude = np.zeros((batch_size, self.n_items), dtype=theano.config.floatX)

		
		for i, sequence in enumerate(sequences):
			user_id, in_seq, target = sequence
			seq_features = np.array(map(lambda x: self._get_features(x, user_id), in_seq))
			X[i, :len(in_seq), :] = seq_features # Copy sequences into X
			mask[i, :len(in_seq)] = 1
			Y[i] = target[0][0] # id of the first and only target
			exclude[i, [j[0] for j in in_seq]] = 1
			pop[i] = self.dataset.item_popularity[target[0][0]] ** self.diversity_bias

		if self.sampling_bias > 0:
			samples = np.array([self._popularity_sample() for i in range(self.effective_sampling)], dtype='int32')
		else:
			samples = np.random.choice(self.n_items, self.effective_sampling).astype('int32')


		return (X, mask.astype(theano.config.floatX), Y, samples, pop.astype(theano.config.floatX), exclude)
