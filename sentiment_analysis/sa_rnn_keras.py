"""
Sentiment analysis by RNN using Keras
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from utils.dataset_utils import get_traintest_tweet
from utils.plot import plot_metrics

def create_model(num_words, embedding_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_words, embedding_dim),
        tf.keras.layers.SimpleRNN(units=256, return_sequences=True),
        tf.keras.layers.SimpleRNN(units=128, return_sequences=True),
        tf.keras.layers.SimpleRNN(units=64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model
    

def sa_rnn_keras_running():
    train_x, val_x, train_y, val_y, num_words, max_len = get_traintest_tweet()
    
    model = create_model(num_words=num_words, embedding_dim=64)
    print("Training...")
    history = model.fit(train_x,
                        train_y,
                        epochs=20,
                        validation_data=(val_x, val_y))
    
    # convert model to ONNX
    model.save('model/SA_RNN_keras.h5')
    print("Model was saved!")
    
    plot_metrics(history, "accuracy", namefig='SA_RNN_keras_acc')
    plot_metrics(history, "loss", namefig='SA_RNN_keras_loss')
    
    