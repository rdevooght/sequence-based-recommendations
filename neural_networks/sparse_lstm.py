import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
# import theano.sparse as sparse
import lasagne
from lasagne import nonlinearities, init
from lasagne.random import get_rng
from lasagne.utils import unroll_scan
from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers.recurrent import Gate
from lasagne.layers import helper
from collections import OrderedDict

def discriminative_cost(predictions, targets):
	pos = T.nnet.categorical_crossentropy(predictions, targets)
	neg = T.log(1 - predictions)
	return pos + neg[:, targets.shape[0]:].sum(axis=-1)


class BlackoutLayer(DenseLayer):
	def __init__(self, incoming, num_units, num_outputs=0.01, **kwargs):
		super(BlackoutLayer, self).__init__(incoming, num_units, **kwargs)
		self._srng = RandomStreams(get_rng().randint(1, 2147462579))
		if num_outputs < 1:
			num_outputs = num_outputs * num_units
		self.num_outputs = int(num_outputs)

	def get_output_for(self, input, deterministic=False, targets=None, samples=None, **kwargs):
		if input.ndim > 2:
			# if the input has more than two dimensions, flatten it into a
			# batch of feature vectors.
			input = input.flatten(2)

		if deterministic:
			activation = T.dot(input, self.W)
			if self.b is not None:
				activation = activation + self.b.dimshuffle('x', 0)
		else:

			if samples is None:
				output_cells = self._srng.choice(a=self.num_units, size=(self.num_outputs,))
			else:
				output_cells = samples

			if targets is not None:
				#output_cells = [x for x in output_cells if x not in targets]
				output_cells = T.concatenate((targets, output_cells))

			activation = T.dot(input, self.W[:,output_cells])
			if self.b is not None:
				activation = activation + self.b[output_cells].dimshuffle('x', 0)

		return self.nonlinearity(activation)


class LSTMLayerOHEInput(MergeLayer):
	r"""
	lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
	ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
	cell=lasagne.layers.Gate(
	W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
	outgate=lasagne.layers.Gate(),
	nonlinearity=lasagne.nonlinearities.tanh,
	cell_init=lasagne.init.Constant(0.),
	hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
	peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
	precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

	A long short-term memory (LSTM) layer.

	Includes optional "peephole connections" and a forget gate.  Based on the
	definition in [1]_, which is the current common definition.  The output is
	computed by

	.. math ::

		i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
			   + w_{ci} \odot c_{t-1} + b_i)\\
		f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
			   + w_{cf} \odot c_{t-1} + b_f)\\
		c_t &= f_t \odot c_{t - 1}
			   + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
		o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
		h_t &= o_t \odot \sigma_h(c_t)

	Parameters
	----------
	incoming : a :class:`lasagne.layers.Layer` instance or a tuple
		The layer feeding into this layer, or the expected input shape.
	num_units : int
		Number of hidden/cell units in the layer.
	ingate : Gate
		Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
		:math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
	forgetgate : Gate
		Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
		:math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
	cell : Gate
		Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
		:math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
	outgate : Gate
		Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
		:math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
	nonlinearity : callable or None
		The nonlinearity that is applied to the output (:math:`\sigma_h`). If
		None is provided, no nonlinearity will be applied.
	cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
		Initializer for initial cell state (:math:`c_0`).
	hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
		Initializer for initial hidden state (:math:`h_0`).
	backwards : bool
		If True, process the sequence backwards and then reverse the
		output again such that the output from the layer is always
		from :math:`x_1` to :math:`x_n`.
	learn_init : bool
		If True, initial hidden values are learned.
	peepholes : bool
		If True, the LSTM uses peephole connections.
		When False, `ingate.W_cell`, `forgetgate.W_cell` and
		`outgate.W_cell` are ignored.
	gradient_steps : int
		Number of timesteps to include in the backpropagated gradient.
		If -1, backpropagate through the entire sequence.
	grad_clipping : float
		If nonzero, the gradient messages are clipped to the given value during
		the backward pass.  See [1]_ (p. 6) for further explanation.
	unroll_scan : bool
		If True the recursion is unrolled instead of using scan. For some
		graphs this gives a significant speed up but it might also consume
		more memory. When `unroll_scan` is True, backpropagation always
		includes the full sequence, so `gradient_steps` must be set to -1 and
		the input sequence length must be known at compile time (i.e., cannot
		be given as None).
	precompute_input : bool
		If True, precompute input_to_hid before iterating through
		the sequence. This can result in a speedup at the expense of
		an increase in memory usage.
	mask_input : :class:`lasagne.layers.Layer`
		Layer which allows for a sequence mask to be input, for when sequences
		are of variable length.  Default `None`, which means no mask will be
		supplied (i.e. all sequences are of the same length).
	only_return_final : bool
		If True, only return the final sequential output (e.g. for tasks where
		a single target value for the entire sequence is desired).  In this
		case, Theano makes an optimization which saves memory.

	References
	----------
	.. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
		   arXiv preprint arXiv:1308.0850 (2013).
	"""
	def __init__(self, incoming, num_units, input_size,
				 ingate=Gate(),
				 forgetgate=Gate(),
				 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
				 outgate=Gate(),
				 nonlinearity=nonlinearities.tanh,
				 cell_init=init.Constant(0.),
				 hid_init=init.Constant(0.),
				 backwards=False,
				 learn_init=False,
				 peepholes=True,
				 gradient_steps=-1,
				 grad_clipping=0,
				 unroll_scan=False,
				 precompute_input=True,
				 mask_input=None,
				 only_return_final=False,
				 **kwargs):

		# This layer inherits from a MergeLayer, because it can have four
		# inputs - the layer input, the mask, the initial hidden state and the
		# inital cell state. We will just provide the layer input as incomings,
		# unless a mask input, inital hidden state or initial cell state was
		# provided.
		incomings = [incoming]
		self.mask_incoming_index = -1
		self.hid_init_incoming_index = -1
		self.cell_init_incoming_index = -1
		if mask_input is not None:
			incomings.append(mask_input)
			self.mask_incoming_index = len(incomings)-1
		if isinstance(hid_init, Layer):
			incomings.append(hid_init)
			self.hid_init_incoming_index = len(incomings)-1
		if isinstance(cell_init, Layer):
			incomings.append(cell_init)
			self.cell_init_incoming_index = len(incomings)-1

		# Initialize parent layer
		super(LSTMLayerOHEInput, self).__init__(incomings, **kwargs)

		# If the provided nonlinearity is None, make it linear
		if nonlinearity is None:
			self.nonlinearity = nonlinearities.identity
		else:
			self.nonlinearity = nonlinearity

		self.learn_init = learn_init
		self.num_units = num_units
		self.input_size = input_size
		self.backwards = backwards
		self.peepholes = peepholes
		self.gradient_steps = gradient_steps
		self.grad_clipping = grad_clipping
		self.unroll_scan = unroll_scan
		self.precompute_input = precompute_input
		self.only_return_final = only_return_final

		if unroll_scan and gradient_steps != -1:
			raise ValueError(
				"Gradient steps must be -1 when unroll_scan is true.")

		# Retrieve the dimensionality of the incoming layer
		input_shape = self.input_shapes[0]

		if unroll_scan and input_shape[1] is None:
			raise ValueError("Input sequence length cannot be specified as "
							 "None when unroll_scan is True")

		#num_inputs = np.prod(input_shape[2:])
		num_inputs = self.input_size

		def add_gate_params(gate, gate_name):
			""" Convenience function for adding layer parameters from a Gate
			instance. """
			return (self.add_param(gate.W_in, (num_inputs, num_units),
								   name="W_in_to_{}".format(gate_name)),
					self.add_param(gate.W_hid, (num_units, num_units),
								   name="W_hid_to_{}".format(gate_name)),
					self.add_param(gate.b, (num_units,),
								   name="b_{}".format(gate_name),
								   regularizable=False),
					gate.nonlinearity)

		# Add in parameters from the supplied Gate instances
		(self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
		 self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

		(self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
		 self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
														 'forgetgate')

		(self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
		 self.nonlinearity_cell) = add_gate_params(cell, 'cell')

		(self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
		 self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

		# If peephole (cell to gate) connections were enabled, initialize
		# peephole connections.  These are elementwise products with the cell
		# state, so they are represented as vectors.
		if self.peepholes:
			self.W_cell_to_ingate = self.add_param(
				ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

			self.W_cell_to_forgetgate = self.add_param(
				forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

			self.W_cell_to_outgate = self.add_param(
				outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

		# Setup initial values for the cell and the hidden units
		if isinstance(cell_init, Layer):
			self.cell_init = cell_init
		else:
			self.cell_init = self.add_param(
				cell_init, (1, num_units), name="cell_init",
				trainable=learn_init, regularizable=False)

		if isinstance(hid_init, Layer):
			self.hid_init = hid_init
		else:
			self.hid_init = self.add_param(
				hid_init, (1, self.num_units), name="hid_init",
				trainable=learn_init, regularizable=False)

	def get_output_shape_for(self, input_shapes):
		# The shape of the input to this layer will be the first element
		# of input_shapes, whether or not a mask input is being used.
		input_shape = input_shapes[0]
		# When only_return_final is true, the second (sequence step) dimension
		# will be flattened
		if self.only_return_final:
			return input_shape[0], self.num_units
		# Otherwise, the shape will be (n_batch, n_steps, num_units)
		else:
			return input_shape[0], input_shape[1], self.num_units

	def get_output_for(self, inputs, **kwargs):
		"""
		Compute this layer's output function given a symbolic input variable

		Parameters
		----------
		inputs : list of theano.TensorType
			`inputs[0]` should always be the symbolic input variable.  When
			this layer has a mask input (i.e. was instantiated with
			`mask_input != None`, indicating that the lengths of sequences in
			each batch vary), `inputs` should have length 2, where `inputs[1]`
			is the `mask`.  The `mask` should be supplied as a Theano variable
			denoting whether each time step in each sequence in the batch is
			part of the sequence or not.  `mask` should be a matrix of shape
			``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
			(length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
			of sequence i)``. When the hidden state of this layer is to be
			pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
			should have length at least 2, and `inputs[-1]` is the hidden state
			to prefill with. When the cell state of this layer is to be
			pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
			should have length at least 2, and `inputs[-1]` is the hidden state
			to prefill with. When both the cell state and the hidden state are
			being pre-filled `inputs[-2]` is the hidden state, while
			`inputs[-1]` is the cell state.

		Returns
		-------
		layer_output : theano.TensorType
			Symbolic output variable.
		"""
		# Retrieve the layer input
		input = inputs[0].astype('int32')
		# Retrieve the mask when it is supplied
		mask = None
		hid_init = None
		cell_init = None
		if self.mask_incoming_index > 0:
			mask = inputs[self.mask_incoming_index]
		if self.hid_init_incoming_index > 0:
			hid_init = inputs[self.hid_init_incoming_index]
		if self.cell_init_incoming_index > 0:
			cell_init = inputs[self.cell_init_incoming_index]

		# Treat all dimensions after the second as flattened feature dimensions
		if input.ndim > 3:
			input = T.flatten(input, 3)

		# Because scan iterates over the first dimension we dimshuffle to
		# (n_time_steps, n_batch, n_features)
		input = input.dimshuffle(1, 0, 2)
		seq_len, num_batch, _ = input.shape

		# Stack input weight matrices into a (num_inputs, 4*num_units)
		# matrix, which speeds up computation
		W_in_stacked = T.concatenate(
			[self.W_in_to_ingate, self.W_in_to_forgetgate,
			 self.W_in_to_cell, self.W_in_to_outgate], axis=1)

		# Same for hidden weight matrices
		W_hid_stacked = T.concatenate(
			[self.W_hid_to_ingate, self.W_hid_to_forgetgate,
			 self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

		# Stack biases into a (4*num_units) vector
		b_stacked = T.concatenate(
			[self.b_ingate, self.b_forgetgate,
			 self.b_cell, self.b_outgate], axis=0)

		if self.precompute_input:
			# Because the input is given for all time steps, we can
			# precompute_input the inputs dot weight matrices before scanning.
			# W_in_stacked is (n_features, 4*num_units). input is then
			# (n_time_steps, n_batch, 4*num_units).
			#input = T.dot(input, W_in_stacked) + b_stacked
			input = W_in_stacked[input, :].sum(axis = -2) + b_stacked

		# At each call to scan, input_n will be (n_time_steps, 4*num_units).
		# We define a slicing function that extract the input to each LSTM gate
		def slice_w(x, n):
			return x[:, n*self.num_units:(n+1)*self.num_units]

		# Create single recurrent computation step function
		# input_n is the n'th vector of the input
		def step(input_n, cell_previous, hid_previous, *args):
			if not self.precompute_input:
				#input_n = T.dot(input_n, W_in_stacked) + b_stacked
				input_n = W_in_stacked[input_n, :].sum(axis = -2) + b_stacked

			# Calculate gates pre-activations and slice
			gates = input_n + T.dot(hid_previous, W_hid_stacked)

			# Clip gradients
			if self.grad_clipping:
				gates = theano.gradient.grad_clip(
					gates, -self.grad_clipping, self.grad_clipping)

			# Extract the pre-activation gate values
			ingate = slice_w(gates, 0)
			forgetgate = slice_w(gates, 1)
			cell_input = slice_w(gates, 2)
			outgate = slice_w(gates, 3)

			if self.peepholes:
				# Compute peephole connections
				ingate += cell_previous*self.W_cell_to_ingate
				forgetgate += cell_previous*self.W_cell_to_forgetgate

			# Apply nonlinearities
			ingate = self.nonlinearity_ingate(ingate)
			forgetgate = self.nonlinearity_forgetgate(forgetgate)
			cell_input = self.nonlinearity_cell(cell_input)

			# Compute new cell value
			cell = forgetgate*cell_previous + ingate*cell_input

			if self.peepholes:
				outgate += cell*self.W_cell_to_outgate
			outgate = self.nonlinearity_outgate(outgate)

			# Compute new hidden unit activation
			hid = outgate*self.nonlinearity(cell)
			return [cell, hid]

		def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
			cell, hid = step(input_n, cell_previous, hid_previous, *args)

			# Skip over any input with mask 0 by copying the previous
			# hidden state; proceed normally for any input with mask 1.
			cell = T.switch(mask_n, cell, cell_previous)
			hid = T.switch(mask_n, hid, hid_previous)

			return [cell, hid]

		if mask is not None:
			# mask is given as (batch_size, seq_len). Because scan iterates
			# over first dimension, we dimshuffle to (seq_len, batch_size) and
			# add a broadcastable dimension
			mask = mask.dimshuffle(1, 0, 'x')
			sequences = [input, mask]
			step_fun = step_masked
		else:
			sequences = input
			step_fun = step

		ones = T.ones((num_batch, 1))
		if not isinstance(self.cell_init, Layer):
			# Dot against a 1s vector to repeat to shape (num_batch, num_units)
			cell_init = T.dot(ones, self.cell_init)

		if not isinstance(self.hid_init, Layer):
			# Dot against a 1s vector to repeat to shape (num_batch, num_units)
			hid_init = T.dot(ones, self.hid_init)

		# The hidden-to-hidden weight matrix is always used in step
		non_seqs = [W_hid_stacked]
		# The "peephole" weight matrices are only used when self.peepholes=True
		if self.peepholes:
			non_seqs += [self.W_cell_to_ingate,
						 self.W_cell_to_forgetgate,
						 self.W_cell_to_outgate]

		# When we aren't precomputing the input outside of scan, we need to
		# provide the input weights and biases to the step function
		if not self.precompute_input:
			non_seqs += [W_in_stacked, b_stacked]

		if self.unroll_scan:
			# Retrieve the dimensionality of the incoming layer
			input_shape = self.input_shapes[0]
			# Explicitly unroll the recurrence instead of using scan
			cell_out, hid_out = unroll_scan(
				fn=step_fun,
				sequences=sequences,
				outputs_info=[cell_init, hid_init],
				go_backwards=self.backwards,
				non_sequences=non_seqs,
				n_steps=input_shape[1])
		else:
			# Scan op iterates over first dimension of input and repeatedly
			# applies the step function
			cell_out, hid_out = theano.scan(
				fn=step_fun,
				sequences=sequences,
				outputs_info=[cell_init, hid_init],
				go_backwards=self.backwards,
				truncate_gradient=self.gradient_steps,
				non_sequences=non_seqs,
				strict=True)[0]

		# When it is requested that we only return the final sequence step,
		# we need to slice it out immediately after scan is applied
		if self.only_return_final:
			hid_out = hid_out[-1]
		else:
			# dimshuffle back to (n_batch, n_time_steps, n_features))
			hid_out = hid_out.dimshuffle(1, 0, 2)

			# if scan is backward reverse the output
			if self.backwards:
				hid_out = hid_out[:, ::-1]

		return hid_out

class GRULayerOHEInput(MergeLayer):
	r"""
	lasagne.layers.recurrent.GRULayer(incoming, num_units,
	resetgate=lasagne.layers.Gate(W_cell=None),
	updategate=lasagne.layers.Gate(W_cell=None),
	hidden_update=lasagne.layers.Gate(
	W_cell=None, lasagne.nonlinearities.tanh),
	hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
	gradient_steps=-1, grad_clipping=0, unroll_scan=False,
	precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

	Gated Recurrent Unit (GRU) Layer

	Implements the recurrent step proposed in [1]_, which computes the output
	by

	.. math ::
		r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
		u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
		c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
		h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t

	Parameters
	----------
	incoming : a :class:`lasagne.layers.Layer` instance or a tuple
		The layer feeding into this layer, or the expected input shape.
	num_units : int
		Number of hidden units in the layer.
	resetgate : Gate
		Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
		:math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
	updategate : Gate
		Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
		:math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
	hidden_update : Gate
		Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
		:math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
	hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
		Initializer for initial hidden state (:math:`h_0`).
	backwards : bool
		If True, process the sequence backwards and then reverse the
		output again such that the output from the layer is always
		from :math:`x_1` to :math:`x_n`.
	learn_init : bool
		If True, initial hidden values are learned.
	gradient_steps : int
		Number of timesteps to include in the backpropagated gradient.
		If -1, backpropagate through the entire sequence.
	grad_clipping : float
		If nonzero, the gradient messages are clipped to the given value during
		the backward pass.  See [1]_ (p. 6) for further explanation.
	unroll_scan : bool
		If True the recursion is unrolled instead of using scan. For some
		graphs this gives a significant speed up but it might also consume
		more memory. When `unroll_scan` is True, backpropagation always
		includes the full sequence, so `gradient_steps` must be set to -1 and
		the input sequence length must be known at compile time (i.e., cannot
		be given as None).
	precompute_input : bool
		If True, precompute input_to_hid before iterating through
		the sequence. This can result in a speedup at the expense of
		an increase in memory usage.
	mask_input : :class:`lasagne.layers.Layer`
		Layer which allows for a sequence mask to be input, for when sequences
		are of variable length.  Default `None`, which means no mask will be
		supplied (i.e. all sequences are of the same length).
	only_return_final : bool
		If True, only return the final sequential output (e.g. for tasks where
		a single target value for the entire sequence is desired).  In this
		case, Theano makes an optimization which saves memory.

	References
	----------
	.. [1] Cho, Kyunghyun, et al: On the properties of neural
	   machine translation: Encoder-decoder approaches.
	   arXiv preprint arXiv:1409.1259 (2014).
	.. [2] Chung, Junyoung, et al.: Empirical Evaluation of Gated
	   Recurrent Neural Networks on Sequence Modeling.
	   arXiv preprint arXiv:1412.3555 (2014).
	.. [3] Graves, Alex: "Generating sequences with recurrent neural networks."
		   arXiv preprint arXiv:1308.0850 (2013).

	Notes
	-----
	An alternate update for the candidate hidden state is proposed in [2]_:

	.. math::
		c_t &= \sigma_c(x_t W_{ic} + (r_t \odot h_{t - 1})W_{hc} + b_c)\\

	We use the formulation from [1]_ because it allows us to do all matrix
	operations in a single dot product.
	"""
	def __init__(self, incoming, num_units, input_size,
				 resetgate=Gate(W_cell=None),
				 updategate=Gate(W_cell=None),
				 hidden_update=Gate(W_cell=None,
									nonlinearity=nonlinearities.tanh),
				 hid_init=init.Constant(0.),
				 backwards=False,
				 learn_init=False,
				 gradient_steps=-1,
				 grad_clipping=0,
				 unroll_scan=False,
				 precompute_input=True,
				 mask_input=None,
				 only_return_final=False,
				 **kwargs):

		# This layer inherits from a MergeLayer, because it can have three
		# inputs - the layer input, the mask and the initial hidden state.  We
		# will just provide the layer input as incomings, unless a mask input
		# or initial hidden state was provided.
		incomings = [incoming]
		self.mask_incoming_index = -1
		self.hid_init_incoming_index = -1
		if mask_input is not None:
			incomings.append(mask_input)
			self.mask_incoming_index = len(incomings)-1
		if isinstance(hid_init, Layer):
			incomings.append(hid_init)
			self.hid_init_incoming_index = len(incomings)-1

		# Initialize parent layer
		super(GRULayerOHEInput, self).__init__(incomings, **kwargs)

		self.learn_init = learn_init
		self.num_units = num_units
		self.input_size = input_size
		self.grad_clipping = grad_clipping
		self.backwards = backwards
		self.gradient_steps = gradient_steps
		self.unroll_scan = unroll_scan
		self.precompute_input = precompute_input
		self.only_return_final = only_return_final

		if unroll_scan and gradient_steps != -1:
			raise ValueError(
				"Gradient steps must be -1 when unroll_scan is true.")

		# Retrieve the dimensionality of the incoming layer
		input_shape = self.input_shapes[0]

		if unroll_scan and input_shape[1] is None:
			raise ValueError("Input sequence length cannot be specified as "
							 "None when unroll_scan is True")

		# Input dimensionality is the output dimensionality of the input layer
		#num_inputs = np.prod(input_shape[2:])
		num_inputs = self.input_size

		def add_gate_params(gate, gate_name):
			""" Convenience function for adding layer parameters from a Gate
			instance. """
			return (self.add_param(gate.W_in, (num_inputs, num_units),
								   name="W_in_to_{}".format(gate_name)),
					self.add_param(gate.W_hid, (num_units, num_units),
								   name="W_hid_to_{}".format(gate_name)),
					self.add_param(gate.b, (num_units,),
								   name="b_{}".format(gate_name),
								   regularizable=False),
					gate.nonlinearity)

		# Add in all parameters from gates
		(self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
		 self.nonlinearity_updategate) = add_gate_params(updategate,
														 'updategate')
		(self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
		 self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

		(self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
		 self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
			 hidden_update, 'hidden_update')

		# Initialize hidden state
		if isinstance(hid_init, Layer):
			self.hid_init = hid_init
		else:
			self.hid_init = self.add_param(
				hid_init, (1, self.num_units), name="hid_init",
				trainable=learn_init, regularizable=False)

	def get_output_shape_for(self, input_shapes):
		# The shape of the input to this layer will be the first element
		# of input_shapes, whether or not a mask input is being used.
		input_shape = input_shapes[0]
		# When only_return_final is true, the second (sequence step) dimension
		# will be flattened
		if self.only_return_final:
			return input_shape[0], self.num_units
		# Otherwise, the shape will be (n_batch, n_steps, num_units)
		else:
			return input_shape[0], input_shape[1], self.num_units

	def get_output_for(self, inputs, **kwargs):
		"""
		Compute this layer's output function given a symbolic input variable

		Parameters
		----------
		inputs : list of theano.TensorType
			`inputs[0]` should always be the symbolic input variable.  When
			this layer has a mask input (i.e. was instantiated with
			`mask_input != None`, indicating that the lengths of sequences in
			each batch vary), `inputs` should have length 2, where `inputs[1]`
			is the `mask`.  The `mask` should be supplied as a Theano variable
			denoting whether each time step in each sequence in the batch is
			part of the sequence or not.  `mask` should be a matrix of shape
			``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
			(length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
			of sequence i)``. When the hidden state of this layer is to be
			pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
			should have length at least 2, and `inputs[-1]` is the hidden state
			to prefill with.

		Returns
		-------
		layer_output : theano.TensorType
			Symbolic output variable.
		"""
		# Retrieve the layer input
		input = inputs[0].astype('int32')
		# Retrieve the mask when it is supplied
		mask = None
		hid_init = None
		if self.mask_incoming_index > 0:
			mask = inputs[self.mask_incoming_index]
		if self.hid_init_incoming_index > 0:
			hid_init = inputs[self.hid_init_incoming_index]

		# Treat all dimensions after the second as flattened feature dimensions
		if input.ndim > 3:
			input = T.flatten(input, 3)

		# Because scan iterates over the first dimension we dimshuffle to
		# (n_time_steps, n_batch, n_features)
		input = input.dimshuffle(1, 0, 2)
		seq_len, num_batch, _ = input.shape

		# Stack input weight matrices into a (num_inputs, 3*num_units)
		# matrix, which speeds up computation
		W_in_stacked = T.concatenate(
			[self.W_in_to_resetgate, self.W_in_to_updategate,
			 self.W_in_to_hidden_update], axis=1)

		# Same for hidden weight matrices
		W_hid_stacked = T.concatenate(
			[self.W_hid_to_resetgate, self.W_hid_to_updategate,
			 self.W_hid_to_hidden_update], axis=1)

		# Stack gate biases into a (3*num_units) vector
		b_stacked = T.concatenate(
			[self.b_resetgate, self.b_updategate,
			 self.b_hidden_update], axis=0)

		if self.precompute_input:
			# precompute_input inputs*W. W_in is (n_features, 3*num_units).
			# input is then (n_batch, n_time_steps, 3*num_units).
			#input = T.dot(input, W_in_stacked) + b_stacked
			input = W_in_stacked[input, :].sum(axis = -2) + b_stacked

		# At each call to scan, input_n will be (n_time_steps, 3*num_units).
		# We define a slicing function that extract the input to each GRU gate
		def slice_w(x, n):
			return x[:, n*self.num_units:(n+1)*self.num_units]

		# Create single recurrent computation step function
		# input__n is the n'th vector of the input
		def step(input_n, hid_previous, *args):
			# Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
			hid_input = T.dot(hid_previous, W_hid_stacked)

			if self.grad_clipping:
				input_n = theano.gradient.grad_clip(
					input_n, -self.grad_clipping, self.grad_clipping)
				hid_input = theano.gradient.grad_clip(
					hid_input, -self.grad_clipping, self.grad_clipping)

			if not self.precompute_input:
				# Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
				#input_n = T.dot(input_n, W_in_stacked) + b_stacked
				input_n = W_in_stacked[input_n, :].sum(axis = -2) + b_stacked

			# Reset and update gates
			resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
			updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
			resetgate = self.nonlinearity_resetgate(resetgate)
			updategate = self.nonlinearity_updategate(updategate)

			# Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
			hidden_update_in = slice_w(input_n, 2)
			hidden_update_hid = slice_w(hid_input, 2)
			hidden_update = hidden_update_in + resetgate*hidden_update_hid
			if self.grad_clipping:
				hidden_update = theano.gradient.grad_clip(
					hidden_update, -self.grad_clipping, self.grad_clipping)
			hidden_update = self.nonlinearity_hid(hidden_update)

			# Compute (1 - u_t)h_{t - 1} + u_t c_t
			hid = (1 - updategate)*hid_previous + updategate*hidden_update
			return hid

		def step_masked(input_n, mask_n, hid_previous, *args):
			hid = step(input_n, hid_previous, *args)

			# Skip over any input with mask 0 by copying the previous
			# hidden state; proceed normally for any input with mask 1.
			hid = T.switch(mask_n, hid, hid_previous)

			return hid

		if mask is not None:
			# mask is given as (batch_size, seq_len). Because scan iterates
			# over first dimension, we dimshuffle to (seq_len, batch_size) and
			# add a broadcastable dimension
			mask = mask.dimshuffle(1, 0, 'x')
			sequences = [input, mask]
			step_fun = step_masked
		else:
			sequences = [input]
			step_fun = step

		if not isinstance(self.hid_init, Layer):
			# Dot against a 1s vector to repeat to shape (num_batch, num_units)
			hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

		# The hidden-to-hidden weight matrix is always used in step
		non_seqs = [W_hid_stacked]
		# When we aren't precomputing the input outside of scan, we need to
		# provide the input weights and biases to the step function
		if not self.precompute_input:
			non_seqs += [W_in_stacked, b_stacked]

		if self.unroll_scan:
			# Retrieve the dimensionality of the incoming layer
			input_shape = self.input_shapes[0]
			# Explicitly unroll the recurrence instead of using scan
			hid_out = unroll_scan(
				fn=step_fun,
				sequences=sequences,
				outputs_info=[hid_init],
				go_backwards=self.backwards,
				non_sequences=non_seqs,
				n_steps=input_shape[1])[0]
		else:
			# Scan op iterates over first dimension of input and repeatedly
			# applies the step function
			hid_out = theano.scan(
				fn=step_fun,
				sequences=sequences,
				go_backwards=self.backwards,
				outputs_info=[hid_init],
				non_sequences=non_seqs,
				truncate_gradient=self.gradient_steps,
				strict=True)[0]

		# When it is requested that we only return the final sequence step,
		# we need to slice it out immediately after scan is applied
		if self.only_return_final:
			hid_out = hid_out[-1]
		else:
			# dimshuffle back to (n_batch, n_time_steps, n_features))
			hid_out = hid_out.dimshuffle(1, 0, 2)

			# if scan is backward reverse the output
			if self.backwards:
				hid_out = hid_out[:, ::-1]

		return hid_out


class VanillaLayerOHEInput(MergeLayer):
	r"""
	lasagne.layers.recurrent.GRULayer(incoming, num_units,
	resetgate=lasagne.layers.Gate(W_cell=None),
	updategate=lasagne.layers.Gate(W_cell=None),
	hidden_update=lasagne.layers.Gate(
	W_cell=None, lasagne.nonlinearities.tanh),
	hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
	gradient_steps=-1, grad_clipping=0, unroll_scan=False,
	precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

	Gated Recurrent Unit (GRU) Layer

	Implements the recurrent step proposed in [1]_, which computes the output
	by

	.. math ::
		r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
		u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
		c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
		h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t

	Parameters
	----------
	incoming : a :class:`lasagne.layers.Layer` instance or a tuple
		The layer feeding into this layer, or the expected input shape.
	num_units : int
		Number of hidden units in the layer.
	resetgate : Gate
		Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
		:math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
	updategate : Gate
		Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
		:math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
	hidden_update : Gate
		Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
		:math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
	hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
		Initializer for initial hidden state (:math:`h_0`).
	backwards : bool
		If True, process the sequence backwards and then reverse the
		output again such that the output from the layer is always
		from :math:`x_1` to :math:`x_n`.
	learn_init : bool
		If True, initial hidden values are learned.
	gradient_steps : int
		Number of timesteps to include in the backpropagated gradient.
		If -1, backpropagate through the entire sequence.
	grad_clipping : float
		If nonzero, the gradient messages are clipped to the given value during
		the backward pass.  See [1]_ (p. 6) for further explanation.
	unroll_scan : bool
		If True the recursion is unrolled instead of using scan. For some
		graphs this gives a significant speed up but it might also consume
		more memory. When `unroll_scan` is True, backpropagation always
		includes the full sequence, so `gradient_steps` must be set to -1 and
		the input sequence length must be known at compile time (i.e., cannot
		be given as None).
	precompute_input : bool
		If True, precompute input_to_hid before iterating through
		the sequence. This can result in a speedup at the expense of
		an increase in memory usage.
	mask_input : :class:`lasagne.layers.Layer`
		Layer which allows for a sequence mask to be input, for when sequences
		are of variable length.  Default `None`, which means no mask will be
		supplied (i.e. all sequences are of the same length).
	only_return_final : bool
		If True, only return the final sequential output (e.g. for tasks where
		a single target value for the entire sequence is desired).  In this
		case, Theano makes an optimization which saves memory.

	References
	----------
	.. [1] Cho, Kyunghyun, et al: On the properties of neural
	   machine translation: Encoder-decoder approaches.
	   arXiv preprint arXiv:1409.1259 (2014).
	.. [2] Chung, Junyoung, et al.: Empirical Evaluation of Gated
	   Recurrent Neural Networks on Sequence Modeling.
	   arXiv preprint arXiv:1412.3555 (2014).
	.. [3] Graves, Alex: "Generating sequences with recurrent neural networks."
		   arXiv preprint arXiv:1308.0850 (2013).

	Notes
	-----
	An alternate update for the candidate hidden state is proposed in [2]_:

	.. math::
		c_t &= \sigma_c(x_t W_{ic} + (r_t \odot h_{t - 1})W_{hc} + b_c)\\

	We use the formulation from [1]_ because it allows us to do all matrix
	operations in a single dot product.
	"""
	def __init__(self, incoming, num_units, input_size,
				 hidden_update=Gate(W_cell=None,
									nonlinearity=nonlinearities.tanh),
				 hid_init=init.Constant(0.),
				 backwards=False,
				 learn_init=False,
				 gradient_steps=-1,
				 grad_clipping=0,
				 unroll_scan=False,
				 precompute_input=True,
				 mask_input=None,
				 only_return_final=False,
				 **kwargs):

		# This layer inherits from a MergeLayer, because it can have three
		# inputs - the layer input, the mask and the initial hidden state.  We
		# will just provide the layer input as incomings, unless a mask input
		# or initial hidden state was provided.
		incomings = [incoming]
		self.mask_incoming_index = -1
		self.hid_init_incoming_index = -1
		if mask_input is not None:
			incomings.append(mask_input)
			self.mask_incoming_index = len(incomings)-1
		if isinstance(hid_init, Layer):
			incomings.append(hid_init)
			self.hid_init_incoming_index = len(incomings)-1

		# Initialize parent layer
		super(VanillaLayerOHEInput, self).__init__(incomings, **kwargs)

		self.learn_init = learn_init
		self.num_units = num_units
		self.input_size = input_size
		self.grad_clipping = grad_clipping
		self.backwards = backwards
		self.gradient_steps = gradient_steps
		self.unroll_scan = unroll_scan
		self.precompute_input = precompute_input
		self.only_return_final = only_return_final

		if unroll_scan and gradient_steps != -1:
			raise ValueError(
				"Gradient steps must be -1 when unroll_scan is true.")

		# Retrieve the dimensionality of the incoming layer
		input_shape = self.input_shapes[0]

		if unroll_scan and input_shape[1] is None:
			raise ValueError("Input sequence length cannot be specified as "
							 "None when unroll_scan is True")

		# Input dimensionality is the output dimensionality of the input layer
		#num_inputs = np.prod(input_shape[2:])
		num_inputs = self.input_size

		def add_gate_params(gate, gate_name):
			""" Convenience function for adding layer parameters from a Gate
			instance. """
			return (self.add_param(gate.W_in, (num_inputs, num_units),
								   name="W_in_to_{}".format(gate_name)),
					self.add_param(gate.W_hid, (num_units, num_units),
								   name="W_hid_to_{}".format(gate_name)),
					self.add_param(gate.b, (num_units,),
								   name="b_{}".format(gate_name),
								   regularizable=False),
					gate.nonlinearity)

		# Add in all parameters from gates
		(self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
		 self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
			 hidden_update, 'hidden_update')

		# Initialize hidden state
		if isinstance(hid_init, Layer):
			self.hid_init = hid_init
		else:
			self.hid_init = self.add_param(
				hid_init, (1, self.num_units), name="hid_init",
				trainable=learn_init, regularizable=False)

	def get_output_shape_for(self, input_shapes):
		# The shape of the input to this layer will be the first element
		# of input_shapes, whether or not a mask input is being used.
		input_shape = input_shapes[0]
		# When only_return_final is true, the second (sequence step) dimension
		# will be flattened
		if self.only_return_final:
			return input_shape[0], self.num_units
		# Otherwise, the shape will be (n_batch, n_steps, num_units)
		else:
			return input_shape[0], input_shape[1], self.num_units

	def get_output_for(self, inputs, **kwargs):
		"""
		Compute this layer's output function given a symbolic input variable

		Parameters
		----------
		inputs : list of theano.TensorType
			`inputs[0]` should always be the symbolic input variable.  When
			this layer has a mask input (i.e. was instantiated with
			`mask_input != None`, indicating that the lengths of sequences in
			each batch vary), `inputs` should have length 2, where `inputs[1]`
			is the `mask`.  The `mask` should be supplied as a Theano variable
			denoting whether each time step in each sequence in the batch is
			part of the sequence or not.  `mask` should be a matrix of shape
			``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
			(length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
			of sequence i)``. When the hidden state of this layer is to be
			pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
			should have length at least 2, and `inputs[-1]` is the hidden state
			to prefill with.

		Returns
		-------
		layer_output : theano.TensorType
			Symbolic output variable.
		"""
		# Retrieve the layer input
		input = inputs[0].astype('int32')
		# Retrieve the mask when it is supplied
		mask = None
		hid_init = None
		if self.mask_incoming_index > 0:
			mask = inputs[self.mask_incoming_index]
		if self.hid_init_incoming_index > 0:
			hid_init = inputs[self.hid_init_incoming_index]

		# Treat all dimensions after the second as flattened feature dimensions
		if input.ndim > 3:
			input = T.flatten(input, 3)

		# Because scan iterates over the first dimension we dimshuffle to
		# (n_time_steps, n_batch, n_features)
		input = input.dimshuffle(1, 0, 2)
		seq_len, num_batch, _ = input.shape

		# Stack input weight matrices into a (num_inputs, 3*num_units)
		# matrix, which speeds up computation
		W_in_stacked = self.W_in_to_hidden_update

		# Same for hidden weight matrices
		W_hid_stacked = self.W_hid_to_hidden_update

		# Stack gate biases into a (3*num_units) vector
		b_stacked = self.b_hidden_update

		if self.precompute_input:
			# precompute_input inputs*W. W_in is (n_features, 3*num_units).
			# input is then (n_batch, n_time_steps, 3*num_units).
			#input = T.dot(input, W_in_stacked) + b_stacked
			input = W_in_stacked[input, :].sum(axis = -2) + b_stacked

		# At each call to scan, input_n will be (n_time_steps, 3*num_units).
		# We define a slicing function that extract the input to each GRU gate
		def slice_w(x, n):
			return x[:, n*self.num_units:(n+1)*self.num_units]

		# Create single recurrent computation step function
		# input__n is the n'th vector of the input
		def step(input_n, hid_previous, *args):
			# Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
			hid_input = T.dot(hid_previous, W_hid_stacked)

			if self.grad_clipping:
				input_n = theano.gradient.grad_clip(
					input_n, -self.grad_clipping, self.grad_clipping)
				hid_input = theano.gradient.grad_clip(
					hid_input, -self.grad_clipping, self.grad_clipping)

			if not self.precompute_input:
				# Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
				#input_n = T.dot(input_n, W_in_stacked) + b_stacked
				input_n = W_in_stacked[input_n, :].sum(axis = -2) + b_stacked

			

			# Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
			hidden_update = input_n + hid_input
			if self.grad_clipping:
				hidden_update = theano.gradient.grad_clip(
					hidden_update, -self.grad_clipping, self.grad_clipping)
			
			return self.nonlinearity_hid(hidden_update)

		def step_masked(input_n, mask_n, hid_previous, *args):
			hid = step(input_n, hid_previous, *args)

			# Skip over any input with mask 0 by copying the previous
			# hidden state; proceed normally for any input with mask 1.
			hid = T.switch(mask_n, hid, hid_previous)

			return hid

		if mask is not None:
			# mask is given as (batch_size, seq_len). Because scan iterates
			# over first dimension, we dimshuffle to (seq_len, batch_size) and
			# add a broadcastable dimension
			mask = mask.dimshuffle(1, 0, 'x')
			sequences = [input, mask]
			step_fun = step_masked
		else:
			sequences = [input]
			step_fun = step

		if not isinstance(self.hid_init, Layer):
			# Dot against a 1s vector to repeat to shape (num_batch, num_units)
			hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

		# The hidden-to-hidden weight matrix is always used in step
		non_seqs = [W_hid_stacked]
		# When we aren't precomputing the input outside of scan, we need to
		# provide the input weights and biases to the step function
		if not self.precompute_input:
			non_seqs += [W_in_stacked, b_stacked]

		if self.unroll_scan:
			# Retrieve the dimensionality of the incoming layer
			input_shape = self.input_shapes[0]
			# Explicitly unroll the recurrence instead of using scan
			hid_out = unroll_scan(
				fn=step_fun,
				sequences=sequences,
				outputs_info=[hid_init],
				go_backwards=self.backwards,
				non_sequences=non_seqs,
				n_steps=input_shape[1])[0]
		else:
			# Scan op iterates over first dimension of input and repeatedly
			# applies the step function
			hid_out = theano.scan(
				fn=step_fun,
				sequences=sequences,
				go_backwards=self.backwards,
				outputs_info=[hid_init],
				non_sequences=non_seqs,
				truncate_gradient=self.gradient_steps,
				strict=True)[0]

		# When it is requested that we only return the final sequence step,
		# we need to slice it out immediately after scan is applied
		if self.only_return_final:
			hid_out = hid_out[-1]
		else:
			# dimshuffle back to (n_batch, n_time_steps, n_features))
			hid_out = hid_out.dimshuffle(1, 0, 2)

			# if scan is backward reverse the output
			if self.backwards:
				hid_out = hid_out[:, ::-1]

		return hid_out
