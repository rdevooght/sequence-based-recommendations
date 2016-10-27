from __future__ import print_function
import numpy as np

def sequence_noise_command_parser(parser):
	parser.add_argument('--n_dropout', help='Dropout probability', default=0., type=float)
	parser.add_argument('--n_swap', help="Probability of swapping two consecutive items", default=0., type=float)
	parser.add_argument('--n_shuf', help="Probability of swapping two random items", default=0., type=float)
	parser.add_argument('--n_shuf_std', help="The distance between the two items to be swapped is drawn from a normal distribution whose std is defined by this parameter", default=5., type=float)
	parser.add_argument('--n_ratings', help='Probability of changing the rating.', default=0., type=float)

def get_sequence_noise(args):
	return SequenceNoise(dropout=args.n_dropout, swap=args.n_swap, ratings_perturb=args.n_ratings, shuf=args.n_shuf, shuf_std=args.n_shuf_std)
	

class SequenceNoise(object):
	def __init__(self, dropout=0., swap=0., ratings_perturb=0., shuf=0., shuf_std=0.):
		super(SequenceNoise, self).__init__()
		self.dropout = dropout
		self.swap = swap
		self.ratings_perturb = ratings_perturb
		self.shuf = shuf
		self.shuf_std = shuf_std

		self._check_param_validity()
		self._set_name()


	def _set_name(self):
		name = []
		if self.dropout > 0:
			name.append("do"+str(self.dropout))

		if self.swap > 0:
			name.append("sw"+str(self.swap))

		if self.ratings_perturb > 0:
			name.append("rp"+str(self.ratings_perturb))

		if self.shuf > 0:
			name.append("sh"+str(self.shuf)+"-"+str(self.shuf_std))

		self.name = "_".join(name)

	def _check_param_validity(self):
		if self.dropout < 0. or self.dropout >= 1.:
			raise ValueError('Dropout should be in [0,1)')
		if self.swap < 0. or self.swap >= 1.:
			raise ValueError('Swapping probability should be in [0,1)')
		if self.ratings_perturb < 0. or self.ratings_perturb >= 1.:
			raise ValueError('Rating perturbation probability should be in [0,1)')

	def __call__(self, sequence_generator):
		"""Recieves a generator of sequences in the form ([(item, rating), (item, rating), ...], user) and generates sequences in the same format,
		after potentially applying dropout, item swapping and ratings modifications.
		"""

		while True:

			sequence, user = next(sequence_generator)

			# Dropout
			if self.dropout > 0.:
				sequence = [i for i in sequence if (np.random.random() >= self.dropout)]
				if len(sequence) < 2:
					continue
			
			# Perturb the order
			if self.swap > 0.:
				i = 0
				while i < len(sequence) - 1:
					if np.random.random() < self.swap:
						tmp = sequence[i]
						sequence[i] = sequence[i+1]
						sequence[i+1] = tmp
						i+=1 # Don't allow to swap twice the same item
					i += 1

			# Shuffle
			if self.shuf > 0.:
				for i in range(len(sequence)):
					if np.random.random() < self.shuf:
						other_item = max(0, min(len(sequence)-1, int(np.random.randn()*self.shuf_std)+i))
						sequence[i], sequence[other_item] = sequence[other_item], sequence[i]

			# Perturb ratings
			if self.ratings_perturb > 0:
				for i in range(len(sequence)):
					if np.random.random() < self.ratings_perturb:
						if np.random.random() < 0.5:
							sequence[i][1] = min(5, sequence[i][1] + 0.5)
						else:
							sequence[i][1] = max(1, sequence[i][1] - 0.5)

			yield sequence, user
	
