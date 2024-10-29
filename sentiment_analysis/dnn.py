
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

from utils.dataset_utils import build_vocab, process_tweet, padded_sequence, get_traintest_tweet

from utils.plot import plot_metrics

def create_model(num_words, embedding_dim, max_len):
    """
    Creates a text classifier model
    
    Args:
        num_words (int): size of the vocabulary for the Embedding layer input
        embedding_dim (int): dimensionality of the Embedding layer output
        max_len (int): length of the input sequences
    
    Returns:
        model (tf.keras Model): the text classifier model
    """
    tf.random.set_seed(123)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def sa_dnn_runner():
    
    train_x, val_x, train_y, val_y, num_words, max_len = get_traintest_tweet()
    
    # Create model
    model = create_model(num_words=num_words, embedding_dim=16, max_len=max_len)
    print('\nThe model is created!\n')
    
    
    print("Training...")
    history = model.fit(train_x,
                        train_y,
                        epochs=20,
                        validation_data=(val_x, val_y))
    
    
    # convert model to ONNX
    model.save('model/SA_DNN.h5')
    print("Model was saved!")
    
    plot_metrics(history, "accuracy", namefig='SA_DNN_acc')
    plot_metrics(history, "loss", namefig='SA_DNN_loss')
    


    