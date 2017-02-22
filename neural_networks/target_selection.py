from __future__ import print_function
import numpy as np
import random

def target_selection_command_parser(parser):
	parser.add_argument('--n_targets', help='Number of targets (Only for RNN with hinge, logit or logsig loss).', default=1, type=int)
	parser.add_argument('--shuffle_targets', help='Instead of picking the next items in the sequence as the target(s), the targets are picked randomly in the remaining sequence.',  action='store_true')
	parser.add_argument('--rand_test_target', help='Use the exact same procedure for target selection during training and testing. Otherwise shuffling and bias are used only during training.',  action='store_true')
	parser.add_argument('--target_bias', help='Popular item are picked as item with a lower probability. Targets are skipped with a probability proportional to (number_of_views)^bias. Set negative bias to avoid this procedure.',  default=-1., type=float)

def get_target_selection(args):
	return SelectTargets(n_targets=args.n_targets, shuffle=args.shuffle_targets, bias=args.target_bias, determinist_test=(not args.rand_test_target))
	

class SelectTargets(object):
	def __init__(self, n_targets=1, shuffle=False, bias=-1, determinist_test=True):
		super(SelectTargets, self).__init__()
		self.n_targets = n_targets
		self.shuffle = shuffle
		self.bias = bias
		self.determinist_test = determinist_test

	@property
	def name(self):
		
		name = "nt"+str(self.n_targets)

		if self.bias >= 0.:
			name += '_tb'+str(self.bias)
		if self.shuffle:
			name += "_shufT"
		return name
			

	def set_dataset(self, dataset):
		
		if self.bias >= 0.:
			pop = np.maximum(1, dataset.item_popularity)
			self.keep_prob = np.power(min(pop) / pop, self.bias) 

	def __call__(self, remaining_sequence, test=False):
		''' Receives the sequence of item that are not read by the RNN and chooses the target(s) among them.
		the test parameter indicates whether this is the training or the testing phase.
		If test is True and self.determinist_test is True, no shuffle nor bias is performed
		'''
		
		if not (test and self.determinist_test):
			if self.shuffle:
				random.shuffle(remaining_sequence)
			if self.bias >= 0.:
				remaining_sequence = [i for i in remaining_sequence if (np.random.random() <= self.keep_prob[i[0]])]

		return remaining_sequence[:min(len(remaining_sequence), self.n_targets)]