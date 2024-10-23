"""
Sentiment analysis by RNN from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from utils.dataset_utils import get_traintest_tweet


class RNN_cell(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, activation='tanh'):
        """ Forward propagation for a a single vanilla RNN cell
        input params:
            - input_dim: embedding dim of each word (shape [1, embedding_dim])
            - output_dim: length of output h_t and y
            - activation: tanh or sigmoid
        output:
            - y: predicted value
            - h_t: current state
        """
        super(RNN_cell, self).__init__()
        
        self.w_hx = np.random.standard_normal((input_dim, output_dim))
        self.w_hh = np.random.standard_normal((output_dim, output_dim))
        self.b_h = np.random.standard_normal((1, output_dim))
        
        
        self.activation = None
        if activation == 'tanh':
            self.activation = tf.keras.activations.tanh()
        elif activation == 'sigmoid':
            self.activation = tf.keras.activations.sigmoid()
            
        
    def call(self, x, h_0):
        h_t = np.dot(h_0, self.w_hh) + np.dot(x, self.w_hx) + self.b_h
        y = self.activation(h_t)
        return y, h_t


class ScratchRNN(tf.keras.Model):
    def __init__(self, num_words, embedding_dim=64, max_len=51, hidden_dim=128):
        """ Define RNN model architecture
        Args:
            - num_words (int): size of the vocabulary for the Embedding layer input
            - embedding_dim (int): dimensionality of the Embedding layer output
            - max_len (int): length of the input sequences
            - hidden_dim (int): number of units in each rnn cell
        Returns:
            - RNN model
        """
        super(ScratchRNN, self).__init__()
        
        self.n_cells = max_len
        self.hidden_dim = hidden_dim
        
        self.embedding = tf.keras.layers.Embedding(num_words, embedding_dim, input_length=max_len)
        
        self.rnn_blocks = []
        for _ in self.n_cells:
            rnn_cell = RNN_cell(input_dim=embedding_dim, output_dim=hidden_dim, activation='tanh')
            self.rnn_blocks.append(rnn_cell)
        self.rnn_blocks.append(rnn_cell)
        
        
    def call(self, x_padded):
        x = self.embedding(x_padded)
        
        h_0 = None
        for i in len(self.RNN_block):
            rnn_cell = RNN_block[i]
            x_t = x[i]
            if i == 0:
                h_0 = np.zeros((1, self.hidden_dim))
            y, h_0 = rnn_cell(x_t, h_0)
        
        return y
    
        
def sa_scratch_rnn_running():
    # train_x_padded, val_x_padded, train_y, val_y = get_traintest_tweet()
    
    n_cells = train_x_padded[0].shape()[0]