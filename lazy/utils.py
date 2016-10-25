from __future__ import division
import numpy as np
import scipy.sparse as ssp

def top_k(values, k, exclude=[]):
	''' Return the indices of the k items with the highest value in the list of values.
	Exclude the ids from the list "exclude".
	'''

	# Put low similarity to viewed items to exclude them from recommendations
	values[exclude] = -np.inf

	return list(np.argpartition(-values, range(k))[:k])

def get_sparse_vector(ids, length, values=None):
	'''Converts a list of ids into a sparse vector of length "length" where the elements corresponding to the ids are given the values in "values".
	If "values" is None, the elements are set to 1.
	'''
	n = len(ids)

	if values is None:
		return ssp.coo_matrix((np.ones(n), (ids,np.zeros(n))), (length, 1)).tocsc()
	else:
		return ssp.coo_matrix((values, (ids,np.zeros(n))), (length, 1)).tocsc()