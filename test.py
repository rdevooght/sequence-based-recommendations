from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import random
import argparse
import re
import glob
import sys
import os
import copy
# import matplotlib.pyplot as plt
from helpers.data_handling import DataHandler
from helpers import evaluation
import helpers.command_parser as parse


def get_file_name(predictor, args):
	return args.dir + re.sub('_ml'+str(args.max_length), '_ml'+str(args.training_max_length), predictor._get_model_filename(args.number_of_batches))

def find_models(predictor, dataset, args):
	if args.method == "UKNN" or args.method == "MM" or args.method == "POP":
		return None

	file = dataset.dirname + "models/" + get_file_name(predictor, args)
	print(file)
	if args.number_of_batches == "*":
		file = np.array(glob.glob(file))

	return file

def save_file_name(predictor, dataset, args):
	if not args.save:
		return None
	else:
		file = re.sub('_ne\*_', '_', dataset.dirname + 'results/' + get_file_name(predictor, args))
		return file


def run_tests(predictor, model_file, dataset, get_full_recommendation_list=False, k=10):
	# Load model
	predictor.load(model_file)
	#predictor.load_last(os.path.dirname(model_file) + '/')
	# Prepare evaluator
	evaluator = evaluation.Evaluator(dataset, k=k)

	if get_full_recommendation_list:
		k = dataset.n_items

	count = 0
	for sequence, user_id in dataset.test_set(epochs=1):
		count += 1
		num_viewed = int(len(sequence) / 2)
		viewed = sequence[:num_viewed]
		goal = [i[0] for i in sequence[num_viewed:]]

		recommendations = predictor.top_k_recommendations(viewed, user_id=user_id, k=k)

		evaluator.add_instance(goal, recommendations)

		if len(goal) == 0:
			raise ValueError
	return evaluator

def print_results(ev, plot=True, file=None, n_batches=None, print_full_rank_comparison=False):
	print('Average precision: ', ev.average_precision())
	print('Average recall: ', ev.average_recall())
	print('Average ndcg: ', ev.average_ndcg())
	print('Percentage of strict success: ', ev.strict_success_percentage())

	goalDistrib = evaluation.DistributionCharacteristics(ev.get_all_goals())
	strictGoalDistrib = evaluation.DistributionCharacteristics(ev.get_strict_goals())
	predictionsDistrib = evaluation.DistributionCharacteristics(ev.get_all_predictions())
	successDistrib = evaluation.DistributionCharacteristics(ev.get_correct_predictions())
	strictSuccessDistrib = evaluation.DistributionCharacteristics(ev.get_correct_strict_predictions())

	# goalDistrib.plot_frequency_distribution()
	# predictionsDistrib.plot_frequency_distribution()
	# successDistrib.plot_frequency_distribution()
	print('movies correctly recommended : ', successDistrib.number_of_movies())
	print('movies correctly recommended (strict) : ', strictSuccessDistrib.number_of_movies())

	# if plot:
	# 	goalDistrib.plot_popularity_distribution()
	# 	predictionsDistrib.plot_popularity_distribution()
	# 	successDistrib.plot_popularity_distribution()

	if file != None:
		if not os.path.exists(os.path.dirname(file)):
			os.makedirs(os.path.dirname(file))
		with open(file, "a") as f:
			f.write("\t".join(map(str, [n_batches, ev.average_precision(), ev.strict_success_percentage(), ev.general_success_percentage(), successDistrib.number_of_movies(), ev.average_recall(), ev.average_ndcg(), ev.success_in_top_items()])) + "\n")
			# f.write("\t".join(map(str, [n_batches, ev.average_precision(), ev.strict_success_percentage(), ev.general_success_percentage(), goalDistrib.number_of_movies(), predictionsDistrib.number_of_movies(), successDistrib.number_of_movies(), strictGoalDistrib.number_of_movies(), strictSuccessDistrib.number_of_movies(), ev.average_recall(), ev.average_ndcg(), ev.average_novelty(), ev.success_in_top_items()])) + "\n")
		if print_full_rank_comparison:
			with open(file+"_full_rank", "a") as f:
				for data in ev.get_rank_comparison():
					f.write("\t".join(map(str, data)) + "\n")
	else:
		print("\t".join(map(str, ['-', ev.average_precision(), ev.strict_success_percentage(), ev.general_success_percentage(), successDistrib.number_of_movies(), ev.average_recall(), ev.average_ndcg(), ev.success_in_top_items()])), file=sys.stderr)
		# print("\t".join(map(str, ['-', ev.average_precision(), ev.strict_success_percentage(), ev.general_success_percentage(), goalDistrib.number_of_movies(), predictionsDistrib.number_of_movies(), successDistrib.number_of_movies(), strictGoalDistrib.number_of_movies(), strictSuccessDistrib.number_of_movies(), ev.average_recall(), ev.average_ndcg(), ev.average_novelty(), ev.success_in_top_items()])), file=sys.stderr)
		if print_full_rank_comparison:
			with open(file+"_full_rank", "a") as f:
				for data in ev.get_rank_comparison():
					f.write("\t".join(map(str, data)) + "\n")

def extract_number_of_epochs(filename):
	m = re.search('_ne([0-9]+(\.[0-9]+)?)_', filename)
	return float(m.group(1))

def get_last_tested_batch(filename):
	'''If the output file exist already, it will look at the content of the file and return the last batch that was tested.
	This is used to avoid testing to times the same model.
	'''
	
	if filename is not None and os.path.isfile(filename):
		with open(filename) as f:
			for line in f:
				pass
			return float(line.split()[0])
	else:
		return 0

def test_command_parser(parser):
	
	parser.add_argument('-d', dest='dataset', help='Directory name of the dataset.', default='', type=str)
	parser.add_argument('-i', dest='number_of_batches', help='Number of epochs, if not set it will compare all the available models', default=-1, type=int)
	parser.add_argument('-k', dest='nb_of_predictions', help='Number of predictions to make. It is the "k" in "prec@k", "rec@k", etc.', default=10, type=int)
	parser.add_argument('--save', help='Save results to a file', action='store_true')
	parser.add_argument('--dir', help='Model directory.', default="", type=str)
	parser.add_argument('--save_rank', help='Save the full comparison of goal and prediction ranking.', action='store_true')

def main():
	
	args = parse.command_parser(parse.predictor_command_parser, test_command_parser)

	args.training_max_length = args.max_length
	# args.max_length = int(DATA_HANDLER.max_length/2)
	if args.number_of_batches == -1:
		args.number_of_batches = "*"

	dataset = DataHandler(dirname=args.dataset)
	predictor = parse.get_predictor(args)
	predictor.prepare_model(dataset)
	file = find_models(predictor, dataset, args)

	if args.number_of_batches == "*" and args.method != "UKNN" and args.method != "MM" and args.method != "POP":
		
		output_file = save_file_name(predictor, dataset, args)

		last_tested_batch = get_last_tested_batch(output_file)
		batches = np.array(map(extract_number_of_epochs, file))
		sorted_ids = np.argsort(batches)
		batches = batches[sorted_ids]
		file = file[sorted_ids]
		for i, f in enumerate(file):
			if batches[i] > last_tested_batch:
				evaluator = run_tests(predictor, f, dataset, get_full_recommendation_list=args.save_rank, k=args.nb_of_predictions)
				print('-------------------')
				print('(',i+1 ,'/', len(file),') results on ' + f)
				print_results(evaluator, plot=False, file=output_file, n_batches=batches[i], print_full_rank_comparison=args.save_rank)
	else:
		evaluator = run_tests(predictor, file, dataset, get_full_recommendation_list=args.save_rank, k=args.nb_of_predictions)
		print_results(evaluator, file=save_file_name(predictor, dataset, args), print_full_rank_comparison=args.save_rank)

if __name__ == '__main__':
    main()