import argparse
from neural_networks.rnn_one_hot import RNNOneHot
from neural_networks.rnn_cluster import RNNCluster
from neural_networks.fism_cluster import FISMCluster
from neural_networks.rnn_margin import RNNMargin
from neural_networks.rnn_sampling import RNNSampling
from lazy.pop import Pop
from lazy.markov_model import MarkovModel
from lazy.user_knn import UserKNN
from neural_networks.stacked_denoising_autoencoder import StackedDenoisingAutoencoder
from factorization.bprmf import BPRMF
from factorization.fism import FISM
from factorization.fossil import Fossil
from factorization.fpmc import FPMC
from word2vec.ltm import LTM
from helpers.early_stopping import early_stopping_command_parser, get_early_stopper
from neural_networks.recurrent_layers import recurrent_layers_command_parser, get_recurrent_layers
from neural_networks.update_manager import update_manager_command_parser, get_update_manager
from neural_networks.sequence_noise import sequence_noise_command_parser, get_sequence_noise
from neural_networks.target_selection import target_selection_command_parser, get_target_selection

def command_parser(*sub_command_parser):
	''' *sub_command_parser should be callables that will add arguments to the command parser
	'''

	parser = argparse.ArgumentParser()

	for scp in sub_command_parser:
		scp(parser)

	args = parser.parse_args()
	return args

def predictor_command_parser(parser):
	parser.add_argument('-m', dest='method', choices=['RNN', 'SDA', 'BPRMF', 'FPMC', 'FISM', 'Fossil', 'LTM', 'UKNN', 'MM', 'POP'],
	 help='Method', default='RNN')
	parser.add_argument('-b', dest='batch_size', help='Batch size', default=16, type=int)
	parser.add_argument('-l', dest='learning_rate', help='Learning rate', default=0.01, type=float)
	parser.add_argument('-r', dest='regularization', help='Regularization (positive for L2, negative for L1)', default=0., type=float)
	parser.add_argument('-g', dest='gradient_clipping', help='Gradient clipping', default=100, type=int)
	parser.add_argument('-H', dest='hidden', help='Number of hidden neurons (for LTM and BPRMF)', default=20, type=int)
	parser.add_argument('-L', dest='layers', help='Layers (for SDA)', default="20", type=str)
	parser.add_argument('--loss', help='Loss function, choose between TOP1, BPR and Blackout (Sampling), or hinge, logit and logsig (multi-targets), or CCE (Categorical cross-entropy)', default='CCE', type=str)
	parser.add_argument('--sampling', help='Number of sample for the computation of the loss in RNNSampling', default=32.0, type=float)
	parser.add_argument('--sampling_bias', help='Sampling bias for cluster methods. 0. means uniform sampling, 1. means proportional to the item frequency', default=0., type=float)
	parser.add_argument('--db', dest='diversity_bias', help='Diversity bias (for RNN with CCE, TOP1, BPR or Blackout loss)', default=0.0, type=float)
	parser.add_argument('--in_do', dest='input_dropout', help='Input dropout (for SDA)', default=0.2, type=float)
	parser.add_argument('--do', dest='dropout', help='Dropout (for SDA)', default=0.5, type=float)
	parser.add_argument('--rf', help='Use rating features.', action='store_true')
	parser.add_argument('--mf', help='Use movie features.', action='store_true')
	parser.add_argument('--uf', help='Use users features.', action='store_true')
	parser.add_argument('--ns', help='Neighborhood size (for UKNN).', default=80, type=int)
	parser.add_argument('--pb', help='Popularity based (for RNNMargin).', action='store_true')
	parser.add_argument('--balance', help='Balance between false positive and false negative error (for RNNMargin).', default=1., type=float)
	parser.add_argument('--min_access', help='Estimation of minimum access probability (for RNNMargin).', default=0.05, type=float)
	parser.add_argument('--k_cf', help='Number of features for the CF factorization (for FPMC).', default=32, type=int)
	parser.add_argument('--k_mc', help='Number of features for the MC factorization (for FPMC).', default=32, type=int)
	parser.add_argument('--init_sigma', help='Sigma of the gaussian initialization (for FPMC)', default=1, type=float)
	parser.add_argument('--fpmc_bias', help='Sampling bias (for FPMC)', default=100., type=float)
	parser.add_argument('--no_adaptive_sampling', help='No adaptive sampling (for FPMC)', action='store_true')
	parser.add_argument('--cooling', help='Simulated annealing', default=1., type=float)
	parser.add_argument('--ltm_damping', help='Temporal damping (for LTM)', default=0.8, type=float)
	parser.add_argument('--ltm_window', help='Window for word2vec (for LTM)', default=5, type=int)
	parser.add_argument('--ltm_no_trajectory', help='Do not use users trajectory in LTM, just use word2vec', action='store_true')
	parser.add_argument('--max_length', help='Maximum length of sequences during training (for RNNs)', default=30, type=int)
	parser.add_argument('--repeated_interactions', help='The model can recommend items with which the user already interacted', action='store_true')
	parser.add_argument('--fism_alpha', help='Alpha parameter in FISM', default=0.2, type=float)
	parser.add_argument('--fossil_order', help='Order of the markov chains in Fossil', default=1, type=int)

	parser.add_argument('--c_sampling', help='Number of sample for the clustering loss. If unset, the same samples are used for the recommendation loss and for the clustering loss.', default=-1, type=int)
	parser.add_argument('--ignore_clusters', help='Don\'t use clusters during test. Useful to observe the influence of clustering', action='store_true')
	parser.add_argument('--clusters', help='Number of clusters. If unset, no clustering is used', default=-1, type=int)
	parser.add_argument('--init_scale', help='Initial scale of the softmax and sigmoid in the clustering method.', default=1., type=float)
	parser.add_argument('--scale_growing_rate', help='Rate of the geometric growth of the sigmoid/softmax scale in the clustering method.', default=1., type=float)
	parser.add_argument('--max_scale', help='Max scale of the softmax and sigmoid in the clustering method.', default=50, type=float)
	parser.add_argument('--csn', help='Cluster selection noise', default=0., type=float)
	parser.add_argument('--cluster_type', choices=['softmax', 'mix', 'sigmoid'], help='Type of clusters. Softmax puts every item in 1 and only 1 cluster. Sigmoid allow puts items in 0 to n clusters. Mix puts items in 1 to n clusters.', default='mix', type=str)

	update_manager_command_parser(parser)
	recurrent_layers_command_parser(parser)
	sequence_noise_command_parser(parser)
	target_selection_command_parser(parser)

def get_predictor(args):
	args.layers = map(int, args.layers.split('-'))

	updater = get_update_manager(args)
	recurrent_layer = get_recurrent_layers(args)
	sequence_noise = get_sequence_noise(args)
	target_selection = get_target_selection(args)

	if args.method == "MF":
		return Factorization()
	elif args.method == "BPRMF":
		return BPRMF(k=args.hidden, reg = args.regularization, learning_rate = args.learning_rate, annealing=args.cooling, init_sigma = args.init_sigma, adaptive_sampling=(not args.no_adaptive_sampling), sampling_bias=args.fpmc_bias)
	elif args.method == "FISM":
		if args.clusters > 0:
			return FISMCluster(h=args.hidden, reg=args.regularization, alpha=args.fism_alpha, loss=args.loss, interactions_are_unique=(not args.repeated_interactions), predict_with_clusters=(not args.ignore_clusters), sampling_bias=args.sampling_bias, sampling=args.sampling, cluster_sampling=args.c_sampling, init_scale=args.init_scale, scale_growing_rate=args.scale_growing_rate, max_scale=args.max_scale, n_clusters=args.clusters, cluster_type=args.cluster_type, updater=updater, target_selection=target_selection, sequence_noise=sequence_noise, recurrent_layer=recurrent_layer, use_ratings_features=args.rf, use_movies_features=args.mf, use_users_features=args.uf, batch_size=args.batch_size)
		else:
			return FISM(k=args.hidden, reg = args.regularization, learning_rate = args.learning_rate, annealing=args.cooling, init_sigma = args.init_sigma, loss=args.loss, alpha=args.fism_alpha)
	elif args.method == "Fossil":
		return Fossil(k=args.hidden, order=args.fossil_order, reg = args.regularization, learning_rate = args.learning_rate, annealing=args.cooling, init_sigma = args.init_sigma, alpha=args.fism_alpha)
	elif args.method == "FPMC":
		return FPMC(k_cf = args.k_cf, k_mc = args.k_mc, reg = args.regularization, learning_rate = args.learning_rate, annealing=args.cooling, init_sigma = args.init_sigma, adaptive_sampling=(not args.no_adaptive_sampling), sampling_bias=args.fpmc_bias)
	elif args.method == "LTM":
		return LTM(k = args.hidden, alpha = args.ltm_damping, window = args.ltm_window, learning_rate=args.learning_rate, use_trajectory=(not args.ltm_no_trajectory))
	elif args.method == "UKNN":
		return UserKNN(neighborhood_size=args.ns)
	elif args.method == "POP":
		return Pop()
	elif args.method == "MM":
		return MarkovModel()
	elif args.method == 'RNN':
		if args.clusters > 0:
			return RNNCluster(interactions_are_unique=(not args.repeated_interactions), max_length=args.max_length, cluster_selection_noise=args.csn, loss=args.loss, predict_with_clusters=(not args.ignore_clusters), sampling_bias=args.sampling_bias, sampling=args.sampling, cluster_sampling=args.c_sampling, init_scale=args.init_scale, scale_growing_rate=args.scale_growing_rate, max_scale=args.max_scale, n_clusters=args.clusters, cluster_type=args.cluster_type, updater=updater, target_selection=target_selection, sequence_noise=sequence_noise, recurrent_layer=recurrent_layer, use_ratings_features=args.rf, use_movies_features=args.mf, use_users_features=args.uf, batch_size=args.batch_size)
		elif args.loss == 'CCE':
			return RNNOneHot(interactions_are_unique=(not args.repeated_interactions), max_length=args.max_length, diversity_bias=args.diversity_bias, regularization=args.regularization, updater=updater, target_selection=target_selection, sequence_noise=sequence_noise, recurrent_layer=recurrent_layer, use_ratings_features=args.rf, use_movies_features=args.mf, use_users_features=args.uf, batch_size=args.batch_size)
		elif args.loss in ['hinge', 'logit', 'logsig']:
			return RNNMargin(interactions_are_unique=(not args.repeated_interactions), loss_function=args.loss, balance = args.balance, popularity_based = args.pb, min_access = args.min_access, target_selection=target_selection, sequence_noise=sequence_noise, recurrent_layer=recurrent_layer, max_length=args.max_length, updater=updater, use_ratings_features=args.rf, use_movies_features=args.mf, use_users_features=args.uf, batch_size=args.batch_size)
		elif args.loss in ['BPR', 'TOP1', 'Blackout']:
			return RNNSampling(interactions_are_unique=(not args.repeated_interactions), loss_function=args.loss, diversity_bias=args.diversity_bias, sampling=args.sampling, sampling_bias=args.sampling_bias, recurrent_layer=recurrent_layer, max_length=args.max_length, updater=updater, target_selection=target_selection, sequence_noise=sequence_noise, use_ratings_features=args.rf, use_movies_features=args.mf, use_users_features=args.uf, batch_size=args.batch_size)
		else:
			raise ValueError('Unknown loss for the RNN model')
	elif args.method == "SDA":
		return StackedDenoisingAutoencoder(interactions_are_unique=(not args.repeated_interactions), layers = args.layers, input_dropout=args.input_dropout, dropout=args.dropout, updater=updater, batch_size=args.batch_size, use_ratings_features=args.rf)


	