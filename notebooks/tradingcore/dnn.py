import tensorflow as tf 
from tensorflow.contrib import rnn

def getDNN (x, LSTMCellSize, keep_prob):
    # We will add two dropout layers and LSTM cells with the number of units as LSTMCellSize.
	multi_cell = tf.contrib.rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(LSTMCellSize), output_keep_prob=keep_prob) for _ in range(2)], state_is_tuple=True)

	# We use the cell to create RNN.
	# Note that outputs is not a tensor, it is a list with one element which is numpy array. 
	outputs, states = rnn.static_rnn(multi_cell, [x], dtype=tf.float32) 

	# hidden layer with sigmoid activation.
    # Size of the first fully connected layer is same with number of units in LSTM cell.
    # Common practice for this type of deep neural networks is that the size of the fully connected layers after LSTM cell
    # is reduced by half in each layer.
	W_fc1 = weight_variable([LSTMCellSize, 100]) 
	b_fc1 = bias_variable([100]) 
	h_fc1 = tf.nn.sigmoid(tf.matmul(outputs[0], W_fc1) + b_fc1) 

    # hidden layer with sigmoid activation.
	W_fc2 = weight_variable([100, 50])
	b_fc2 = bias_variable([50])
	h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2) 

	# dropout layer
	drop2 = tf.nn.dropout(h_fc2, keep_prob) 

	# hidden layer with sigmoid activation.
	W_fc3 = weight_variable([50, 25])
	b_fc3 = bias_variable([25])
	h_fc3 = tf.nn.sigmoid(tf.matmul(drop2, W_fc3) + b_fc3) 

	# hidden layer with sigmoid activation
	W_fc4 = weight_variable([25, 7])
	b_fc4 = bias_variable([7])
	h_fc4 = tf.nn.sigmoid(tf.matmul(h_fc3, W_fc4) + b_fc4) 

	# hidden layer with sigmoid activation
	W_fc5 = weight_variable([7, 7])
	b_fc5 = bias_variable([7])
	h_fc5 = tf.nn.sigmoid(tf.matmul(h_fc4, W_fc5) + b_fc5) 

	finalLayerWeights = tf.Variable(tf.random_normal((7, 1))) 
	finalLayerBiases = tf.Variable(tf.random_normal([1])) 

	# Linear activation for the output layer
	# we are predicting a continuous variable
	pred = tf.matmul(h_fc5, finalLayerWeights) + finalLayerBiases
	
	return pred

def weight_variable(shape): 
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial) 
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial) 