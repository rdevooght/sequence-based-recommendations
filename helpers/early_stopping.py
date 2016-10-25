from __future__ import print_function
import numpy as np

def early_stopping_command_parser(parser):
	parser.add_argument('--es_m', dest='early_stopping_method', choices=['WorstTimesX', 'StopAfterN', 'None'], help='Early stopping method', default='None')
	parser.add_argument('--es_n', help='N parameter (for StopAfterN)', default=5, type=int)
	parser.add_argument('--es_x', help='X parameter (for WorstTimesX)', default=2., type=float)
	parser.add_argument('--es_min_wait', help='Mininum wait before stopping (for WorstTimesX)', default=1., type=float)
	parser.add_argument('--es_LiB', help='Lower is better for validation score.', action='store_true')

def get_early_stopper(args):
	if args.early_stopping_method == 'StopAfterN':
		return StopAfterN(n = args.es_n, higher_is_better=(not args.es_LiB))
	elif args.early_stopping_method == 'WorstTimesX':
		return WaitWorstCaseTimesX(x = args.es_x, min_wait=args.es_min_wait, higher_is_better=(not args.es_LiB))
	else:
		return None

class EarlyStopperBase(object):
	def __init__(self, higher_is_better=True):
		super(EarlyStopperBase, self).__init__()

		self.higher_is_better = higher_is_better

	def __call__(self, epochs, val_costs):

		if not self.higher_is_better:
			val_costs = [-i for i in val_costs]

		return self.decideStopping(epochs, val_costs)

	def decideStopping(self, epochs, val_costs):
		pass

class StopAfterN(EarlyStopperBase):
	''' Stops after N consecutively non improving cost
	'''
	def __init__(self, n=3, **kwargs):
		super(StopAfterN, self).__init__(**kwargs)

		self.n = n

	def decideStopping(self, epochs, val_costs):

		if len(val_costs) <= self.n:
			return False

		for i in range(self.n):
			if val_costs[-1-i] > val_costs[-2-i]:
				return False

		return True


class WaitWorstCaseTimesX(EarlyStopperBase):
	''' Stops if the number of epochs since the best cost is X times larger than the maximum number of epochs between two consecutive best.
	'''

	def __init__(self, x=2., min_wait=1., **kwargs):
		super(WaitWorstCaseTimesX, self).__init__(**kwargs)

		self.x = x
		self.min_wait = min_wait

	def decideStopping(self, epochs, val_costs):

		# find longest wait between two best scores
		last_best = val_costs[0]
		last_best_epoch = epochs[0]
		longest_wait = 0
		for epoch, cost in zip(epochs[1:], val_costs[1:]):
			if cost > last_best:
				wait = epoch - last_best_epoch
				last_best_epoch = epoch
				last_best = cost
				if wait > longest_wait:
					longest_wait = wait

		current_wait = epochs[-1] - last_best_epoch

		if longest_wait == 0:
			return current_wait > self.min_wait

		print('current wait : ', round(current_wait, 3), ' longest wait : ', round(longest_wait, 3), ' ratio : ', current_wait/longest_wait, ' / ', self.x)
		
		return current_wait > max(self.min_wait, longest_wait*self.x)