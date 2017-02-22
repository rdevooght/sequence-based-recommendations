import lasagne
import numpy as np
import theano
import theano.tensor as T

class SparseLayer(lasagne.layers.DenseLayer):
	
	def __init__(self, incoming, **kwargs):
		super(SparseLayer, self).__init__(incoming, **kwargs)

	def get_output_for(self, input, **kwargs):
		
		activation = theano.sparse.structured_dot(input, self.W)
		if self.b is not None:
			activation = activation + self.b
		return self.nonlinearity(activation)