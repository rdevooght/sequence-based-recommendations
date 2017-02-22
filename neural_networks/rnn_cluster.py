from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import os
import sys
import random
from bisect import bisect
from time import time
import rnn_base as rnn
from sparse_lstm import *
from helpers import evaluation
from theano.sandbox.rng_mrg import MRG_RandomStreams


class RNNCluster(rnn.RNNBase):
	"""RNNCluster combines sampling-based RNN with item clustering.

	Parameters
	----------
	n_clusters: int
		Number of clusters

	loss: "Blackout", "CCE", "BPR" or "BPRelu"
		Determines the loss function, among:
			- BPR, as used in "Session-based Recommendations with Recurrent Neural Networks", Hidasi, B. et al., 2016
			- TOP1, defined in "Session-based Recommendations with Recurrent Neural Networks", Hidasi, B. et al., 2016
			- Blackout, discriminative loss function defined in "BlackOut: Speeding up Recurrent Neural Network Language Models With Very Large Vocabularies", Ji, S. et al., 2015 (equation 6)
			- BPRelu, approximation of BPR based on relu/hinge non-linearities
			- CCE, categorical cross-entropy computed on the set of samples

	cluster_type: "mix", "softmax" or "sigmoid"
		Determines whether items can belong to multiple clusters.
			- mix, items belong to at least one cluster, possibly many.
			- softmax, items belong to one and only one cluster.
			- sigmoid, items belong to zero, one or multiple clusters.

	sampling: int
		Number of samples.

	cluster_sampling: int
		If cluster_sampling > 0, the recommendation loss and the clustering loss use different samples.
		In that case, cluster_sampling is the number of samples used by the clustering loss.

	sampling_bias: float
		Items are sampled with a probability proportional to their frequency to the power of the sampling_bias.

	predict_with_clusters: bool
		Set to false during testing if you want to ignore the clustering.

	cluster_selection_noise: float
		If cluster_selection_noise > 0, a random gaussian noise (whose std is cluster_selection_noise) is added to the cluster selection output during training.
		Can help to explore a large number of clusters.
	
	init_scale: float
		Initial scale of the softmax and sigmoid functions used in the cluster selection process.

	scale_growing_rate: float
		After each training epoch, the scale of the softmax and sigmoid functions is multiplied by the scale_growing_rate.

	max_scale: float
		Maximum allowed scale.

	See classes SequenceNoise, RecurrentLayers, SelectTargets and update manager for options common to the other RNN methods.
	"""

	def __init__(self, n_clusters=10, loss="Blackout", cluster_type='mix', sampling=100, cluster_sampling=-1, sampling_bias=0., predict_with_clusters=True, cluster_selection_noise=0., init_scale=1., scale_growing_rate=1., max_scale=50, **kwargs):
		super(RNNCluster, self).__init__(**kwargs)
		
		self.n_clusters = n_clusters
		self.init_scale = np.cast[theano.config.floatX](init_scale)
		self.effective_scale = np.cast[theano.config.floatX](init_scale)
		self.scale_growing_rate = np.cast[theano.config.floatX](scale_growing_rate)
		self.max_scale = np.cast[theano.config.floatX](max_scale)
		self.cluster_type = cluster_type
		self.sampling_bias = sampling_bias
		self.loss = loss
		self.cluster_selection_noise = cluster_selection_noise

		self.predict_with_clusters = predict_with_clusters

		if self.loss == "Blackout":
			self._loss = self._blackout_loss
		elif self.loss == 'lin':
			self._loss = self._lin_loss
		elif self.loss == 'BPRelu':
			self._loss = self._BPRelu_loss
		elif self.loss == 'BPR':
			self._loss = self._BPR_loss
		elif self.loss == 'TOP1':
			self._loss = self._TOP1_loss
		elif self.loss == 'CCE':
			self._loss = self._cce_loss
		else: 
			raise ValueError('Unknown cluster loss')
			

		self.n_samples = int(sampling)
		self.n_cluster_samples = int(cluster_sampling)

		self._srng = MRG_RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
		

		self.name = "RNN Cluster with categorical cross entropy"

		self.metrics = {'recall': {'direction': 1}, 
			'cluster_recall': {'direction': 1}, 
			'sps': {'direction': 1}, 
			'cluster_sps': {'direction': 1}, 
			'ignored_items': {'direction': -1}, 
			'assr': {'direction': 1}, 
			'cluster_use': {'direction': 1}, 
			'cluster_use_std': {'direction': -1}, 
			'cluster_size': {'direction': 1}
		}

	def _get_model_filename(self, epochs):
		'''Return the name of the file to save the current model
		'''
		filename = "rnn_clusters"+str(self.n_clusters)+"_sc"+str(self.init_scale)

		if self.scale_growing_rate != 1.:
			filename += "-"+str(self.scale_growing_rate)+"-"+str(self.max_scale)

		filename+="_"
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

		filename += "_c" + self.loss
			
		return filename+"_"+self._common_filename(epochs)

	def _blackout_loss(self, predictions, n_targets):
		targets = np.arange(n_targets)
		predictions = T.nnet.softmax(predictions)
		pos = T.nnet.categorical_crossentropy(predictions, targets)
		neg = T.log(1 - predictions)
		return pos - neg[:, targets.shape[0]:].sum(axis=-1)

	def _cce_loss(self, predictions, n_targets):
		targets = np.arange(n_targets)
		predictions = T.nnet.softmax(predictions)
		pos = T.nnet.categorical_crossentropy(predictions, targets)
		return pos

	def _lin_loss(self, predictions, n_targets):
		neg = predictions[:, n_targets:].sum(axis=-1)
		pos = T.diag(predictions)
		return neg - pos

	def _BPR_loss(self, predictions, n_targets):
		diff = (predictions - T.diag(predictions).dimshuffle([0,'x']))[:, n_targets:]
		return -(T.log(T.nnet.sigmoid(-diff))).mean(axis=-1)

	def _BPRelu_loss(self, predictions, n_targets):
		diff = (predictions - T.diag(predictions).dimshuffle([0,'x']))[:, n_targets:]
		return lasagne.nonlinearities.leaky_rectify(diff+0.5).mean(axis=-1)

	def _TOP1_loss(self, predictions, n_targets):
		diff = (predictions - T.diag(predictions).dimshuffle([0,'x']))[:, n_targets:]
		reg = T.sqr(predictions[:, n_targets:])
		return (T.nnet.sigmoid(diff) + T.nnet.sigmoid(reg)).mean(axis=-1)

	def _create_ini_clusters(self):
		c = 0.1 * np.random.randn(self.n_items, self.n_clusters)
		# c = -2 * np.random.random((self.n_items, self.n_clusters)) - 1
		# for i, j in enumerate(np.random.choice(self.n_clusters, self.n_items)):
		# 	c[i,j] *= -1

		# print(np.round(c[:5, :], 2))
		return c.astype(theano.config.floatX)

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

		
		# Theano tensor for the targets
		self.target = T.ivector('target_output')
		self.exclude = T.fmatrix('excluded_items')
		self.samples = T.ivector('samples')
		self.cluster_samples = T.ivector('cluster_samples')
		
		self.user_representation_layer = l_recurrent
		
		# The sliced output is then passed through linear layer to obtain the right output size
		self.l_out = BlackoutLayer(l_recurrent, num_units=self.n_items, num_outputs=self.n_samples, nonlinearity=None, W=lasagne.init.GlorotUniform())

		# lasagne.layers.get_output produces a variable for the output of the net
		network_output = lasagne.layers.get_output(self.l_out, targets = self.target, samples=self.samples)

		# loss function
		self.cost = self._loss(network_output,self.batch_size).mean()


		# Cluster learning
		self.T_scale = theano.shared(self.effective_scale)
		scaled_softmax = lambda x: lasagne.nonlinearities.softmax(x*self.T_scale)

		self.cluster_selection_layer = lasagne.layers.DenseLayer(l_recurrent, b=None, num_units=self.n_clusters, nonlinearity=None)
		cluster_selection = lasagne.layers.get_output(self.cluster_selection_layer)
		if self.cluster_selection_noise > 0.:
			cluster_selection = cluster_selection + self._srng.normal(cluster_selection.shape, avg=0.0, std=self.cluster_selection_noise)
		cluster_selection = scaled_softmax(cluster_selection)

		self.cluster_repartition = theano.shared(self._create_ini_clusters())
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
		self.train_function = theano.function([self.l_in.input_var, self.l_mask.input_var, self.target, self.samples, self.cluster_samples, self.exclude], self.cost, updates=updates, allow_input_downcast=True, name="Train_function", on_unused_input='ignore')
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
			theano_predict_function = theano.function([self.l_in.input_var, self.l_mask.input_var], [user_representation, cluster_selection], allow_input_downcast=True, name="Predict_function", on_unused_input='ignore')

			def cluster_predict_function(sequence, mask, k, exclude):
				u, c = theano_predict_function(sequence, mask)
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
			theano_predict_function = theano.function([self.l_in.input_var, self.l_mask.input_var], user_representation, allow_input_downcast=True, name="Predict_function", on_unused_input='ignore')

			def no_cluster_predict_function(sequence, mask, k, exclude):
				u = theano_predict_function(sequence, mask)
				scores = u[0].dot(self.l_out.W.get_value(borrow=True)) + self.l_out.b.get_value(borrow=True)

				scores[exclude] = -np.inf

				# find top k according to output
				return list(np.argpartition(-scores, range(k))[:k]), self.n_items

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

		theano_test_function = theano.function([self.l_in.input_var, self.l_mask.input_var, self.target, self.samples, self.cluster_samples, self.exclude], [items_score1, items_score2, cluster_selection, used_items.sum()], allow_input_downcast=True, name="Test_function", on_unused_input='ignore')

		def precision_test_function(theano_inputs):
			k = 10
			scores1, scores2, c_select, n_used_items = theano_test_function(*theano_inputs)
			ids1 = np.argpartition(-scores1, range(k), axis=-1)[0, :k]
			ids2 = np.argpartition(-scores2, range(k), axis=-1)[0, :k]
			
			return ids1, ids2, c_select, n_used_items

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
		exclude = np.zeros((batch_size, self.n_items), dtype=theano.config.floatX)

		
		for i, sequence in enumerate(sequences):
			user_id, in_seq, target = sequence
			seq_features = np.array(map(lambda x: self._get_features(x, user_id), in_seq))
			X[i, :len(in_seq), :] = seq_features # Copy sequences into X
			mask[i, :len(in_seq)] = 1
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

		return (X, mask.astype(theano.config.floatX), Y, samples, cluster_samples, exclude)

	def _compute_validation_metrics(self, metrics):
		clusters = np.zeros(self.n_clusters, dtype="int")
		used_items = []
		ev = evaluation.Evaluator(self.dataset, k=10)
		ev_clusters = evaluation.Evaluator(self.dataset, k=10)
		for batch, goal in self._gen_mini_batch(self.dataset.validation_set(epochs=1), test=True):
			pred1, pred2, cl, i = self.test_function(batch)
			ev.add_instance(goal, pred1)
			ev_clusters.add_instance(goal, pred2)
			clusters[cl] += 1
			used_items.append(i)
		
		if self.cluster_type == 'softmax':
			ignored_items = 0
			cluster_size = np.histogram(self.cluster_repartition.get_value(borrow=True).argmax(axis=1), bins=range(self.n_clusters+1))[0].tolist()
		elif self.cluster_type == 'mix':
			ignored_items = 0
			sig_clusters = self.cluster_repartition.get_value(borrow=True) > 0.
			softmax_clusters = self.cluster_repartition.get_value(borrow=True).argmax(axis=1)
			for i in range(self.n_items):
				sig_clusters[i, softmax_clusters[i]] = True
			cluster_size = sig_clusters.sum(axis=0)
		else:
			ignored_items = (self.cluster_repartition.get_value(borrow=True).max(axis=1) < 0.).sum()
			cluster_size = (self.cluster_repartition.get_value(borrow=True) > 0.).sum(axis=0)
		
		metrics['recall'].append(ev.average_recall())
		metrics['cluster_recall'].append(ev_clusters.average_recall())
		metrics['sps'].append(ev.sps())
		metrics['cluster_sps'].append(ev_clusters.sps())
		metrics['assr'].append(self.n_items / np.mean(used_items))
		metrics['ignored_items'].append(ignored_items)
		metrics['cluster_use'].append(clusters)
		metrics['cluster_use_std'].append(np.std(clusters))
		metrics['cluster_size'].append(cluster_size)

		return metrics

	def _print_progress(self, iterations, epochs, start_time, train_costs, metrics, validation_metrics):
		'''Print learning progress in terminal
		'''
		print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
		print("Last train cost : ", train_costs[-1])
		for m in self.metrics.keys():
			print(m, ': ', metrics[m][-1])
			if m in validation_metrics:
				print('Best ', m, ': ', max(np.array(metrics[m])*self.metrics[m]['direction'])*self.metrics[m]['direction'])
		print('-----------------')

		# Print on stderr for easier recording of progress
		print(iterations, epochs, time() - start_time, train_costs[-1], metrics['sps'][-1], metrics['cluster_sps'][-1], metrics['recall'][-1], metrics['cluster_recall'][-1], metrics['assr'][-1], metrics['ignored_items'][-1], metrics['cluster_use_std'][-1], file=sys.stderr)

	def prepare_tests(self):
		'''Take the soft clustering and make actual clusters.
		'''
		cluster_membership = self.cluster_repartition.get_value(borrow=True)
		item_embeddings = self.l_out.W.get_value(borrow=True)
		item_bias = self.l_out.b.get_value(borrow=True)
		self.clusters = [[] for i in range(self.n_clusters)]
		for i in range(cluster_membership.shape[0]):
			no_cluster = True
			best_cluster = 0
			best_val = cluster_membership[i, 0]
			for j in range(self.n_clusters):
				if cluster_membership[i,j] > 0:
					self.clusters[j].append(i)
					no_cluster = False
				elif cluster_membership[i,j] > best_val:
					best_val = cluster_membership[i,j]
					best_cluster = j
			if no_cluster:
				self.clusters[best_cluster].append(i)

		self.clusters = [np.array(c) for c in self.clusters]
		self.clusters_reverse_index = []
		for c in self.clusters:
			self.clusters_reverse_index.append({c[j]: j for j in range(len(c))})
		self.clusters_embeddings = [item_embeddings[:, c] for c in self.clusters]
		self.clusters_bias = [item_bias[c] for c in self.clusters]

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
		X = np.zeros((1, self.max_length, self._input_size()), dtype=self._input_type) # input of the RNN
		X[0, :len(max_length_seq), :] = np.array(map(lambda x: self._get_features(x, user_id), max_length_seq))
		mask = np.zeros((1, self.max_length)) # mask of the input (to deal with sequences of different length)
		mask[0, :len(max_length_seq)] = 1

		# Run RNN
		if self.interactions_are_unique:
			should_exclude = [i[0] for i in sequence]
		else:
			should_exclude = []
		should_exclude.extend(exclude)
		return self.predict_function(X, mask.astype(theano.config.floatX), k, should_exclude)

	def save(self, filename):
		'''Save the parameters of a network into a file
		'''
		print('Save model in ' + filename)
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))
		param = lasagne.layers.get_all_param_values(self.l_out)
		param.append(self.cluster_repartition.get_value(borrow=True))
		param.append([p.get_value(borrow=True) for p in self.cluster_selection_layer.get_params()])
		f = file(filename, 'wb')
		cPickle.dump(param,f,protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		
	def load(self, filename):
		'''Load parameters values form a file
		'''
		f = file(filename, 'rb')
		param = cPickle.load(f)
		f.close()
		lasagne.layers.set_all_param_values(self.l_out, [i.astype(theano.config.floatX) for i in param[:-2]])
		self.cluster_repartition.set_value(param[-2])
		for p, v in zip(self.cluster_selection_layer.get_params(), param[-1]):
			p.set_value(v)

		self.prepare_tests()
