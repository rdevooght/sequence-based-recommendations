from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import scipy.sparse as sp
import theano.sparse
import lasagne
import cPickle
import os
import sys
import random
from time import time
from rnn_cluster import RNNCluster
from sparse_lstm import *
from helpers import evaluation
from helpers.sparse_layer import SparseLayer

class FISMCluster(RNNCluster):
	"""FISMCluster combines FISM with item clustering.

	Parameters
	----------

	h: int
		Size of the embedding.

	alpha: float
		Exponant of the normalization term in FISM

	reg: float
		Regularization coefficient. If reg > 0, L2 regularization is used, otherwise L1 regularization is used with coef -reg.

	FISMCluster is built on top of RNNCluster, all the parameters associated to the clustering are described in RNNCluster.
	"""
	def __init__(self, h=100, alpha=0.5, reg=0.00025, max_length=np.inf, **kwargs):
		super(FISMCluster, self).__init__(max_length=np.inf, **kwargs)
		
		self.n_hidden = h
		self.alpha = alpha
		self.reg = reg
		self.target_selection.shuffle = True
		self.name = "FISM Cluster with categorical cross entropy"
		self.recurrent_layer.name = ""

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "fism_clusters"+str(self.n_clusters)+"_sc"+str(self.init_scale)

		if self.scale_growing_rate != 1.:
			filename += "-"+str(self.scale_growing_rate)+"-"+str(self.max_scale)

		filename += "_h"+ str(self.n_hidden) + "_a" + str(self.alpha) +"_"
		if self.sampling_bias > 0.:
			filename += "p" + str(self.sampling_bias)
		filename += "s"+str(self.n_samples)

		if self.n_cluster_samples > 0:
			filename += "_"
			if self.sampling_bias > 0.:
				filename += "p" + str(self.sampling_bias)
			filename += "cs"+str(self.n_cluster_samples)

		if self.cluster_type == 'softmax':
			filename += "_softmax"
		elif self.cluster_type == 'mix':
			filename += "_mix"

		if self.cluster_selection_noise > 0.:
			filename += '_n' + str(self.cluster_selection_noise)

		if self.reg != 0.:
			filename += '_r' + str(self.reg)

		filename += "_c" + self.loss
			
		return filename+"_"+self._common_filename(epochs)

	def _prepare_networks(self, n_items):
		''' Prepares the building blocks of the RNN, but does not compile them:
		self.l_in : input layer
		self.l_mask : mask of the input layer
		self.target : target of the network
		self.l_out : output of the network
		self.cost : cost function
		'''
	   
		self.n_items = n_items
		
		# Theano tensor for the targets
		input_var = theano.sparse.csr_matrix('input_var')
		self.target = T.ivector('target_output')
		self.exclude = T.fmatrix('excluded_items')
		self.samples = T.ivector('samples')
		self.cluster_samples = T.ivector('cluster_samples')
		
		# The input is composed of to parts : the on-hot encoding of the movie, and the features of the movie
		self.l_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.n_items), input_var=input_var)
		
		l_user_rep = SparseLayer(self.l_in, num_units=self.n_hidden, nonlinearity=None, b=None)

		self.user_representation_layer = l_user_rep

		# The sliced output is then passed through linear layer to obtain the right output size
		self.l_out = BlackoutLayer(l_user_rep, num_units=self.n_items, num_outputs=self.n_samples, nonlinearity=None, W=lasagne.init.GlorotUniform())

		# lasagne.layers.get_output produces a variable for the output of the net
		network_output = lasagne.layers.get_output(self.l_out, targets = self.target, samples=self.samples)

		# loss function
		self.cost = self._loss(network_output,self.batch_size).mean()
		if self.reg > 0.:
			self.cost += self.reg * lasagne.regularization.regularize_network_params(self.l_out, lasagne.regularization.l2)
		elif self.reg < 0.:
			self.cost -= self.reg * lasagne.regularization.regularize_network_params(self.l_out, lasagne.regularization.l1)


		# Cluster learning
		self.T_scale = theano.shared(self.effective_scale)
		scaled_softmax = lambda x: lasagne.nonlinearities.softmax(x*self.T_scale)

		self.cluster_selection_layer = lasagne.layers.DenseLayer(l_user_rep, b=None, num_units=self.n_clusters, nonlinearity=None)
		cluster_selection = lasagne.layers.get_output(self.cluster_selection_layer)
		if self.cluster_selection_noise > 0.:
			cluster_selection = cluster_selection + self._srng.normal(cluster_selection.shape, avg=0.0, std=self.cluster_selection_noise)
		cluster_selection = scaled_softmax(cluster_selection)

		self.cluster_repartition = theano.shared((0.1 * np.random.randn(self.n_items, self.n_clusters)).astype(theano.config.floatX))
		if self.cluster_type == 'softmax':
			target_and_samples_clusters = scaled_softmax(self.cluster_repartition[T.concatenate([self.target, self.cluster_samples]), :])
		elif self.cluster_type == 'mix':
			target_and_samples_clusters = scaled_softmax(self.cluster_repartition[T.concatenate([self.target, self.cluster_samples]), :]) + \
				T.nnet.sigmoid(self.T_scale*self.cluster_repartition[T.concatenate([self.target, self.cluster_samples]), :])
		else:
			target_and_samples_clusters = T.nnet.sigmoid(self.T_scale*self.cluster_repartition[T.concatenate([self.target, self.cluster_samples]), :])
		cluster_score = cluster_selection.dot(target_and_samples_clusters.T)

		self.cost_clusters = self._loss(cluster_score, self.batch_size).mean()

	def _compile_train_function(self):
		''' Compile self.train. 
		self.train recieves a sequence and a target for every steps of the sequence, 
		compute error on every steps, update parameter and return global cost (i.e. the error).
		'''
		print("Compiling train...")
		# Compute AdaGrad updates for training
		all_params = lasagne.layers.get_all_params(self.l_out, trainable=True)
		updates = self.updater(self.cost, all_params)

		params_clusters = self.cluster_selection_layer.get_params(trainable=True)
		params_clusters.append(self.cluster_repartition)
		updates.update(self.updater(self.cost_clusters, params_clusters))
		# Compile network
		self.train_function = theano.function([self.l_in.input_var, self.target, self.samples, self.cluster_samples, self.exclude], self.cost, updates=updates, allow_input_downcast=True, name="Train_function", on_unused_input='ignore')
		print("Compilation done.")

	def _get_hard_clusters(self):
		if self.cluster_type == 'softmax':
			return lasagne.nonlinearities.softmax(100. * self.cluster_repartition)
		elif self.cluster_type == 'mix':
			# Clipping is used to avoid the sum of sigmoid and softmax to produce a cluster indicator of 2
			return (lasagne.nonlinearities.softmax(100. * self.cluster_repartition) + T.nnet.sigmoid(100. * self.cluster_repartition)).clip(0,1)
		else:
			return T.nnet.sigmoid(100. * self.cluster_repartition)

	def _compile_predict_function(self):
		''' Compile self.predict, the deterministic rnn that output the prediction at the end of the sequence
		'''
		print("Compiling predict...")
		if self.predict_with_clusters:
			cluster_selection = lasagne.layers.get_output(self.cluster_selection_layer, deterministic=True)[0, :].argmax()
			user_representation = lasagne.layers.get_output(self.user_representation_layer, deterministic=True)
			theano_predict_function = theano.function([self.l_in.input_var], [user_representation, cluster_selection], allow_input_downcast=True, name="Predict_function", on_unused_input='ignore')

			def cluster_predict_function(sequence, k, exclude):
				u, c = theano_predict_function(sequence)
				c = int(c)
				scores = u[0].dot(self.clusters_embeddings[c]) + self.clusters_bias[c]

				cluster_index_exclude = []
				for i in exclude:
					if i in self.clusters_reverse_index[c]:
						cluster_index_exclude.append(self.clusters_reverse_index[c][i])
				scores[cluster_index_exclude] = -np.inf

				# find top k according to output
				effective_k = min(k, len(self.clusters[c]))
				return list(self.clusters[c][np.argpartition(-scores, range(effective_k))[:effective_k]]), len(self.clusters[c])

			self.predict_function = cluster_predict_function
		else:
			items_score = lasagne.nonlinearities.softmax(lasagne.layers.get_output(self.l_out, deterministic=True))

			user_representation = lasagne.layers.get_output(self.user_representation_layer, deterministic=True)
			theano_predict_function = theano.function([self.l_in.input_var], user_representation, allow_input_downcast=True, name="Predict_function", on_unused_input='ignore')

			def no_cluster_predict_function(sequence, k, exclude):
				u = theano_predict_function(sequence)
				scores = u[0].dot(self.l_out.W.get_value(borrow=True)) + self.l_out.b.get_value(borrow=True)

				scores[exclude] = -np.inf

				# find top k according to output
				return list(np.argpartition(-scores, range(k))[:k]), self.n_items

			# theano_predict_function = theano.function([self.l_in.input_var], items_score, allow_input_downcast=True, name="Predict_function", on_unused_input='ignore')

			# def no_cluster_predict_function(sequence, k, exclude):
			# 	scores = theano_predict_function(sequence)[0]
			# 	scores[exclude] = -np.inf

			# 	# find top k according to output
			# 	return list(np.argpartition(-scores, range(k))[:k]), self.n_items

			self.predict_function = no_cluster_predict_function

		print("Compilation done.")

	def _compile_test_function(self):
		''' Compile self.test_function, the deterministic rnn that output the precision@10
		'''
		print("Compiling test...")
		
		items_score1 = lasagne.nonlinearities.softmax(lasagne.layers.get_output(self.l_out, deterministic=True))
		
		cluster_selection = lasagne.layers.get_output(self.cluster_selection_layer, deterministic=True)[0, :].argmax()
		items_clusters = self._get_hard_clusters()
		used_items = items_clusters[:,cluster_selection]
		items_score2 = items_score1 * used_items

		if self.interactions_are_unique:
			items_score1 *= (1 - self.exclude)
			items_score2 *= (1 - self.exclude)

		theano_test_function = theano.function([self.l_in.input_var, self.target, self.samples, self.cluster_samples, self.exclude], [items_score1, items_score2, cluster_selection, used_items.sum()], allow_input_downcast=True, name="Test_function", on_unused_input='ignore')

		def precision_test_function(theano_inputs):
			k = 10
			scores1, scores2, c_select, n_used_items = theano_test_function(*theano_inputs)
			ids1 = np.argpartition(-scores1, range(k), axis=-1)[0, :k]
			ids2 = np.argpartition(-scores2, range(k), axis=-1)[0, :k]
			
			return ids1, ids2, c_select, n_used_items

		self.test_function = precision_test_function

		print("Compilation done.")

	def _prepare_input(self, sequences):
		''' Sequences is a list of [user_id, input_sequence, targets]
		'''

		batch_size = len(sequences)

		# Shape return variables
		X = sp.lil_matrix((batch_size, self.n_items), dtype=theano.config.floatX)
		Y = np.zeros((batch_size,), dtype='int32') # output target
		exclude = np.zeros((batch_size, self.n_items), dtype=theano.config.floatX)

		
		for i, sequence in enumerate(sequences):
			user_id, in_seq, target = sequence
			for j in in_seq:
				X[i, j[0]] = 1./len(in_seq)**self.alpha
			Y[i] = target[0][0] # id of the first and only target
			exclude[i, [j[0] for j in in_seq]] = 1

		if self.sampling_bias > 0.:
			samples = np.array([self._popularity_sample() for i in range(self.n_samples)], dtype='int32')
			if self.n_cluster_samples > 0:
				cluster_samples = np.array([self._popularity_sample() for i in range(self.n_cluster_samples)], dtype='int32')
			else:
				cluster_samples = samples
		else:
			samples = np.random.choice(self.n_items, self.n_samples).astype('int32')
			if self.n_cluster_samples > 0:
				cluster_samples = np.random.choice(self.n_items, self.n_cluster_samples).astype('int32')
			else:
				cluster_samples = samples

		# scale
		if not hasattr(self, '_last_epoch'):
			self._last_epoch = self.dataset.training_set.epochs
		else:
			if self.dataset.training_set.epochs > self._last_epoch+1 and self.scale_growing_rate != 1.:
				self.effective_scale *= self.scale_growing_rate ** int(self.dataset.training_set.epochs - self._last_epoch)
				self._last_epoch += int(self.dataset.training_set.epochs - self._last_epoch)
				print("New scale: ", self.effective_scale)
				self.T_scale.set_value(self.effective_scale)

		return (X.tocsr(), Y, samples, cluster_samples, exclude)

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
		X = sp.lil_matrix((1, self.n_items), dtype=theano.config.floatX)
		for j in sequence:
			X[0, j[0]] = 1./len(sequence)**self.alpha

		# Run RNN
		if self.interactions_are_unique:
			should_exclude = [i[0] for i in sequence]
		else:
			should_exclude = []
		should_exclude.extend(exclude)
		return self.predict_function(X.tocsr(), k, should_exclude)

	