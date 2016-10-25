from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne
from sparse_lstm import *

def recurrent_layers_command_parser(parser):
	parser.add_argument('--r_t', dest='recurrent_layer_type', choices=['LSTM', 'GRU'], help='Type of recurrent layer', default='GRU')
	parser.add_argument('--r_l', help="Layers' size, (eg: 100-50-50)", default="32", type=str)
	parser.add_argument('--r_bi', help='Bidirectional layers.', action='store_true')

def get_recurrent_layers(args):
	return RecurrentLayers(layer_type=args.recurrent_layer_type, layers=map(int, args.r_l.split('-')), bidirectional=args.r_bi)
	

class RecurrentLayers(object):
	def __init__(self, layer_type="LSTM", layers=[32], bidirectional=False, grad_clipping=100):
		super(RecurrentLayers, self).__init__()
		self.layer_type = layer_type
		self.layers = layers
		self.bidirectional = bidirectional
		self.grad_clip=grad_clipping
		self.set_name()

	def set_name(self):

		self.name = ""
		if self.bidirectional:
			self.name += "b"+self.layer_type+"_"
		elif self.layer_type != "LSTM":
			self.name += self.layer_type+"_"
		
		self.name += "gc"+str(self.grad_clip)+"_h"+('-'.join(map(str,self.layers)))


	def __call__(self, input_layer, mask_layer, true_input_size=None, only_return_final=True):

		orf = False
		prev_layer = input_layer
		for i, h in enumerate(self.layers):
			if i == len(self.layers) - 1:
				orf = only_return_final
			prev_layer = self.get_one_layer(prev_layer, mask_layer, h, true_input_size, orf)

			true_input_size = None # Second layer is always densely encoded

		return prev_layer



	def get_one_layer(self, input_layer, mask_layer, n_hidden, true_input_size, only_return_final):
		if self.bidirectional:
			forward = self.get_unidirectional_layer(input_layer, mask_layer, n_hidden, true_input_size, only_return_final, backwards=False)
			backward = self.get_unidirectional_layer(input_layer, mask_layer, n_hidden, true_input_size, only_return_final, backwards=True)
			return lasagne.layers.ConcatLayer([forward, backward], axis = -1)
		else:
			return self.get_unidirectional_layer(input_layer, mask_layer, n_hidden, true_input_size, only_return_final, backwards=False)

	def get_unidirectional_layer(self, input_layer, mask_layer, n_hidden, true_input_size, only_return_final, backwards=False):
		if true_input_size is not None:
			if self.layer_type == "LSTM":
				return LSTMLayerOHEInput(
					input_layer, n_hidden, true_input_size, mask_input=mask_layer, grad_clipping=self.grad_clip,
					learn_init=True, only_return_final=only_return_final, backwards=backwards)
			elif self.layer_type == "GRU":
				return GRULayerOHEInput(
					input_layer, n_hidden, true_input_size, mask_input=mask_layer, grad_clipping=self.grad_clip,
					learn_init=True, only_return_final=only_return_final, backwards=backwards)
			else:
				raise ValueError('Unknown layer type')
		else:
			if self.layer_type == "LSTM":
				return lasagne.layers.LSTMLayer(
					input_layer, n_hidden, mask_input=mask_layer, grad_clipping=self.grad_clip,
					learn_init=True, only_return_final=only_return_final, backwards=backwards)
			elif self.layer_type == "GRU":
				return lasagne.layers.GRULayer(
					input_layer, n_hidden, mask_input=mask_layer, grad_clipping=self.grad_clip,
					learn_init=True, only_return_final=only_return_final, backwards=backwards)
			else:
				raise ValueError('Unknown layer type')
