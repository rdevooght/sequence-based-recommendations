from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne
from sparse_lstm import *

def recurrent_layers_command_parser(parser):
	parser.add_argument('--r_t', dest='recurrent_layer_type', choices=['LSTM', 'GRU', 'Vanilla'], help='Type of recurrent layer', default='GRU')
	parser.add_argument('--r_l', help="Layers' size, (eg: 100-50-50)", default="50", type=str)
	parser.add_argument('--r_bi', help='Bidirectional layers.', action='store_true')
	parser.add_argument('--r_emb', help='Add an embedding layer before the RNN. Takes the size of the embedding as parameter, a size<1 means no embedding layer.', type=int, default=0)

def get_recurrent_layers(args):
	return RecurrentLayers(layer_type=args.recurrent_layer_type, layers=map(int, args.r_l.split('-')), bidirectional=args.r_bi, embedding_size=args.r_emb)
	

class RecurrentLayers(object):
	def __init__(self, layer_type="LSTM", layers=[32], bidirectional=False, embedding_size=0, grad_clipping=100):
		super(RecurrentLayers, self).__init__()
		self.layer_type = layer_type
		self.layers = layers
		self.bidirectional = bidirectional
		self.embedding_size = embedding_size
		self.grad_clip=grad_clipping
		self.set_name()

	def set_name(self):

		self.name = ""
		if self.bidirectional:
			self.name += "b"+self.layer_type+"_"
		elif self.layer_type != "LSTM":
			self.name += self.layer_type+"_"
		
		self.name += "gc"+str(self.grad_clip)+"_"
		if self.embedding_size > 0:
			self.name += "e"+str(self.embedding_size)
		self.name += "h"+('-'.join(map(str,self.layers)))


	def __call__(self, input_layer, mask_layer, true_input_size=None, only_return_final=True):

		if true_input_size is None and self.embedding_size > 0:
			raise ValueError('Embedding layer only works with sparse inputs')

		if self.embedding_size > 0:
			in_int32 = lasagne.layers.ExpressionLayer(input_layer, lambda x: x.astype('int32')) # change type of input
			l_emb = lasagne.layers.flatten(lasagne.layers.EmbeddingLayer(in_int32, input_size=true_input_size, output_size=self.embedding_size), outdim=3)
			l_rec = self.get_recurrent_layers(l_emb, mask_layer, true_input_size=None, only_return_final=only_return_final)
		else:
			l_rec = self.get_recurrent_layers(input_layer, mask_layer, true_input_size=true_input_size, only_return_final=only_return_final)

		return l_rec


	def get_recurrent_layers(self, input_layer, mask_layer, true_input_size=None, only_return_final=True):

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
				layer = LSTMLayerOHEInput
			elif self.layer_type == "GRU":
				layer = GRULayerOHEInput
			elif self.layer_type == "Vanilla":
				layer = VanillaLayerOHEInput
			else:
				raise ValueError('Unknown layer type')

			return layer(input_layer, n_hidden, true_input_size, mask_input=mask_layer, grad_clipping=self.grad_clip,
				learn_init=True, only_return_final=only_return_final, backwards=backwards)
		else:
			if self.layer_type == "LSTM":
				layer = lasagne.layers.LSTMLayer
			elif self.layer_type == "GRU":
				layer = lasagne.layers.GRULayer
			elif self.layer_type == "Vanilla":
				layer = lasagne.layers.RecurrentLayer
			else:
				raise ValueError('Unknown layer type')

			return layer(input_layer, n_hidden, mask_input=mask_layer, grad_clipping=self.grad_clip,
				learn_init=True, only_return_final=only_return_final, backwards=backwards)
