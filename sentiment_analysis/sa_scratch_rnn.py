"""
Sentiment analysis by RNN from scratch
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from utils.dataset_utils import get_traintest_tweet


class RNN_cell(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        """ Forward propagation for a a single vanilla RNN cell
        input params:
            - input_dim: embedding dim of each word (shape [1, embedding_dim])
            - output_dim: length of output h_t and y
        output:
            - y: predicted value
            - h_t: current state
        """
        super(RNN_cell, self).__init__()
        
        self.tanh = tf.keras.activations.tanh
        self.sigmoid = tf.keras.activations.sigmoid
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.w_hx = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer="random_normal",
            trainable=True,
            name="Whx",
        )
        
        self.w_hh = self.add_weight(
            shape=(self.output_dim, self.output_dim),
            initializer="random_normal",
            trainable=True,
            name="Whh",
        )
        
        self.b_h = self.add_weight(
            shape=(1, self.output_dim),
            initializer="zeros",
            trainable=True,
            name="b_h",
        )
        
        self.w_hy = self.add_weight(
            shape=(self.output_dim, 1),
            initializer="random_normal",
            trainable=True,
            name="w_hy",
        )
        
        self.b_y = self.add_weight(
            shape=(1, 1),
            initializer="zeros",
            trainable=True,
            name="b_y",
        )
        
    def call(self, x, h_0):
        h_t = tf.matmul(h_0, self.w_hh) + tf.matmul(x, self.w_hx) + self.b_h
        h_t = self.tanh(h_t)
        y = tf.matmul(h_t, self.w_hy) + self.b_y
        y = self.sigmoid(y)
        return y, h_t


class ScratchRNNLayer(tf.keras.Model):
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
        super(ScratchRNNLayer, self).__init__()
        
        self.n_cells = max_len
        self.hidden_dim = hidden_dim
        
        self.input_layer = tf.keraslayers.Input(shape=(None, 32, 32))
        
        self.embedding = tf.keras.layers.Embedding(num_words, embedding_dim, input_length=max_len)
        
        self.rnn_blocks = []
        for _ in range(self.n_cells):
            rnn_cell = RNN_cell(input_dim=embedding_dim, output_dim=hidden_dim)
            self.rnn_blocks.append(rnn_cell)
        self.rnn_blocks.append(rnn_cell)
        
        
    def call(self, x_padded):
        x_padded = self.input_layer(x_padded)
        x = self.embedding(x_padded)
        
        batch_size = tf.shape(x)[0]
        h_t = tf.zeros((batch_size, self.hidden_dim))
        
        for i in range(self.n_cells):
            rnn_cell = self.rnn_blocks[i]
            x_t = x[:, i, :]
            y, h_t = rnn_cell(x_t, h_t)
        
        return y
    
        
def sa_scratch_rnn_running():
    # train_x_padded, val_x_padded, train_y, val_y = get_traintest_tweet()
    
    # n_cells = train_x_padded[0].shape()[0]
    
    model = ScratchRNNLayer(num_words=1000)
    model.build(input_shape=(None, 32, 32))
    print(model.summary())
    