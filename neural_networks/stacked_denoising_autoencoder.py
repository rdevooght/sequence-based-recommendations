from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import random
import re
import glob
from time import time
from .rnn_base import RNNBase

def log_softmax(x):
	xdev = x - x.max(1, keepdims=True)
	return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
	return -T.sum(targets * log_predictions, axis=1)

class StackedDenoisingAutoencoder(RNNBase):
	"""Base for Feed forward neural networks object.
	"""
	def __init__(self, layers=[20], input_dropout=0.2, dropout=0.5, **kwargs):
		super(StackedDenoisingAutoencoder, self).__init__(**kwargs)
		
		self.layers = layers
		self.input_dropout = input_dropout
		self.dropout = dropout

		self.name = "Stacked Denoising Autoencoder"


	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "sda_bs"+str(self.batch_size)+"_ne"+str(epochs)
		filename += "_h"+('-'.join(map(str,self.layers)))
		filename += "_" + self.updater.name
		if not self.use_ratings_features:
			filename += "_nf"
		if self.use_ratings_features:
			filename += "_rf"
		return filename

	def top_k_recommendations(self, sequence, user_id=None, k=10, exclude=None, **kwargs):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		# Compile network if needed
		if not hasattr(self, 'predict_function'):
			self._compile_predict_function()

		# Prepare RNN input
		X = np.zeros((1, self._input_size())) # input of the RNN
		X[0, :] = self._one_hot_encoding([i[0] for i in sequence])

		# Run RNN
		output = self.predict_function(X.astype(theano.config.floatX))[0]

		# Put low similarity to viewed items to exclude them from recommendations
		output[[i[0] for i in sequence]] = -np.inf
		output[exclude] = -np.inf

		# find top k according to output
		return list(np.argpartition(-output, range(k))[:k])

	def _prepare_networks(self, n_items):
		''' Prepares the building blocks of the RNN, but does not compile them:
		self.l_in : input layer
		self.target : target of the network
		self.l_out : output of the network
		self.cost : cost function
		'''

		self.n_items = n_items
		
		# The input is composed of to parts : the on-hot encoding of the movie, and the features of the movie
		self.l_in = lasagne.layers.InputLayer(shape=(self.batch_size, self._input_size()))
		# hidden_layer = lasagne.layers.dropout(self.l_in, p=self.input_dropout)
		hidden_layer = self.l_in
		
		# Build hidden layers
		for l in self.layers:
			hidden_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=l)
			if self.dropout:
				hidden_layer = lasagne.layers.dropout(hidden_layer, p=self.dropout)

		# The sliced output is then passed through linear layer to obtain the right output size
		self.l_out = lasagne.layers.DenseLayer(hidden_layer, num_units=self.n_items, nonlinearity=lasagne.nonlinearities.sigmoid)

		# lasagne.layers.get_output produces a variable for the output of the net
		network_output = lasagne.layers.get_output(self.l_out)

		# loss function
		self.targets = T.fmatrix('multiple_target_output')
		self.theano_inputs = [self.l_in.input_var, self.targets]

		self.cost = T.sqr(network_output - self.targets).mean()

	def _compile_predict_function(self):
		''' Compile self.predict, the deterministic rnn that output the prediction at the end of the sequence
		'''
		print("Compiling...")
		deterministic_output = lasagne.layers.get_output(self.l_out, deterministic=True)
		self.predict_function = theano.function([self.l_in.input_var], deterministic_output, allow_input_downcast=True)
		print("Compilation done.")

	def _compile_test_function(self):
		''' Compile self.test_function, the deterministic rnn that output the precision@10
		'''
		print("Compiling test...")
		deterministic_output = lasagne.layers.get_output(self.l_out, deterministic=True)
		if self.interactions_are_unique:
			deterministic_output *= (1 - self.l_in.input_var)
		theano_test_function = theano.function(self.theano_inputs, deterministic_output, allow_input_downcast=True, name="Test_function", on_unused_input='ignore')
		
		def test_function(theano_inputs, k=10):
			output = theano_test_function(*theano_inputs)
			ids = np.argpartition(-output, range(k), axis=-1)[0, :k]
			
			return ids

		self.test_function = test_function

	def _gen_mini_batch(self, sequence_generator, test=False, **kwargs):
		''' Takes a sequence generator and produce a mini batch generator.
		The mini batch have a size defined by self.batch_size, and have format of the input layer of the rnn.

		Assuming that the length of the sequence is bigger than the size of the batch, each batch is created based on one sequence.
		'''

		while True:

			# Shape return variables
			X = np.zeros((self.batch_size, self._input_size())) # input of the RNN
			Y = np.zeros((self.batch_size, self._input_size())) # Target of the RNN
			
			for j in range(self.batch_size):

				sequence, user_id = next(sequence_generator)
				if not test:
					X[j,:] = self._one_hot_encoding([i[0] for i in sequence if (np.random.random() >= self.input_dropout)])
					Y[j, :] = self._one_hot_encoding([i[0] for i in sequence])
					yield (X.astype(theano.config.floatX),Y.astype(theano.config.floatX))
				else:
					X[j, :] = self._one_hot_encoding([i[0] for i in sequence[:len(sequence)/2]])
					Y[j, :] = self._one_hot_encoding(sequence[len(sequence)/2][0])
					yield (X.astype(theano.config.floatX),Y.astype(theano.config.floatX)), [i[0] for i in sequence[len(sequence)/2:]]

	def _one_hot_encoding(self, ids):
		ohe = np.zeros(self._input_size())
		ohe[ids] = 1
		return ohe
	
	def _input_size(self):
		''' Returns the number of input neurons
		'''
		return self.n_items
	
