
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

from utils.dataset_utils import setup_data, load_tweets, process_tweet


def build_vocab(corpus):
    ''' Map each word in each tweet to an interger (an index)
    Input:
        - corpus (list) : the corpus
    Output:
        - vocab (dict): {'words_in_corpus' : 'integer_values'}
    '''
    
    vocab = {
        '': 0,
        '[UNK]': 1
    }
    
    i = 2
    for tweet in corpus:
        for word in tweet:
            if word not in vocab:
                vocab[word] = i
                i += 1
    return vocab


def max_length(training_x, validation_x):
    """Computes the length of the longest tweet in the training and validation sets.

    Args:
        training_x (list): The tweets in the training set.
        validation_x (list): The tweets in the validation set.

    Returns:
        int: Length of the longest tweet.
    """
    max_len = max([len(x) for x in training_x + validation_x])
    
    return max_len


def padded_sequence(tweet, vocab_dict, max_len, unk_token='[UNK]'):
    """transform sequences of words into padded sequences of numbers

    Args:
        tweet (list): A single tweet encoded as a list of strings.
        vocab_dict (dict): Vocabulary.
        max_len (int): Length of the longest tweet.
        unk_token (str, optional): Unknown token. Defaults to '[UNK]'.

    Returns:
        list: Padded tweet encoded as a list of int.
    """
    
    unk_id = vocab_dict[unk_token]
    
    int_words = []
    for x in tweet:
        if x in vocab_dict:
            int_words.append(vocab_dict[x])
        else:
            int_words.append(unk_id)
    padded_tensor = int_words + [0] * (max_len - len(int_words))
    return padded_tensor
    
    
def relu(x):
    '''Relu activation function implementation
    Input: 
        - x (numpy array)
    Output:
        - activation (numpy array): input with negative values set to zero
    '''
    activation = np.maximum(x, 0)    
    return activation
    

def sigmoid(x):
    '''Sigmoid activation function implementation
    Input: 
        - x (numpy array)
    Output:
        - activation (numpy array)
    '''
    activation = 1 / (1 + np.exp(-x))
    return activation    


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


def plot_metrics(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric.title())
    plt.legend([metric, f'val_{metric}'])
    plt.savefig(f'exp/SA_dnn_{metric}.jpg')
    plt.show()
    

def sa_dnn_runner():
    
    setup_data()

    all_positive_tweets, all_negative_tweets = load_tweets()
    print(f"The number of positive tweets: {len(all_positive_tweets)}")
    print(f"The number of negative tweets: {len(all_negative_tweets)}")
    
    tweet_number = 4
    print('\nPositive tweet example:')
    print(all_positive_tweets[tweet_number])
    print('\nNegative tweet example:')
    print(all_negative_tweets[tweet_number])
    
    # Process all the tweets: tokenize the string, remove tickers, handles, punctuation and stopwords, stem the words
    all_positive_tweets_processed = [process_tweet(tweet) for tweet in all_positive_tweets]
    all_negative_tweets_processed = [process_tweet(tweet) for tweet in all_negative_tweets]
    
    
    # Example of processed tweet:
    tweet_number = 4
    print('\nPositive processed tweet example:')
    print(all_positive_tweets_processed[tweet_number])
    print('\nNegative processed tweet example:')
    print(all_negative_tweets_processed[tweet_number])
    
    # Split positive set into validation and training
    val_pos = all_positive_tweets_processed[4000:]
    train_pos = all_positive_tweets_processed[:4000]
    # Split negative set into validation and training
    val_neg = all_negative_tweets_processed[4000:]
    train_neg = all_negative_tweets_processed[:4000]

    train_x = train_pos + train_neg 
    val_x  = val_pos + val_neg

    # Set the labels for the training and validation set (1 for positive, 0 for negative)
    train_y = [[1] for _ in train_pos] + [[0] for _ in train_neg]
    val_y  = [[1] for _ in val_pos] + [[0] for _ in val_neg]

    print(f"\nThere are {len(train_x)} sentences for training.")
    print(f"There are {len(train_y)} labels for training.\n")
    print(f"There are {len(val_x)} sentences for validation.")
    print(f"There are {len(val_y)} labels for validation.")

    
    # Assign each unique word in corpus to an unique integer
    vocab = build_vocab(train_x)
    num_words = len(vocab)

    print(f"\nVocabulary contains {num_words} words\n")
    
    
    # Find the largest value of the length of every sentences (or max length)
    max_len = max_length(train_x, val_x)
    print(f'The length of the longest tweet is {max_len} tokens.')
    
    # Pad the sequence to make sure they are in the same length
    train_x_padded = [padded_sequence(tweet, vocab, max_len) for tweet in train_x]
    val_x_padded = [padded_sequence(tweet, vocab, max_len) for tweet in val_x]
    
    # Create model
    model = create_model(num_words=num_words, embedding_dim=16, max_len=max_len)
    print('\nThe model is created!\n')
    
    # Prepare data for training
    train_x_prepared = np.array(train_x_padded)
    val_x_prepared = np.array(val_x_padded)
    
    train_y_prepared = np.array(train_y)
    val_y_prepared = np.array(val_y)
    
    print('The data is prepared for training!\n')
    
    print("Training...")
    history = model.fit(train_x_prepared,
                        train_y_prepared,
                        epochs=20,
                        validation_data=(val_x_prepared, val_y_prepared))
    
    
    # convert model to ONNX
    model.save('model/SA_DNN.h5')
    print("Model was saved!")
    
    plot_metrics(history, "accuracy")
    plot_metrics(history, "loss")
    

def get_prediction_from_tweet(tweet, model, max_len=51):
    
    all_positive_tweets, all_negative_tweets = load_tweets()
    
    all_positive_tweets_processed = [process_tweet(tweet) for tweet in all_positive_tweets]
    all_negative_tweets_processed = [process_tweet(tweet) for tweet in all_negative_tweets]
    
    val_pos = all_positive_tweets_processed[4000:]
    train_pos = all_positive_tweets_processed[:4000]
    
    val_neg = all_negative_tweets_processed[4000:]
    train_neg = all_negative_tweets_processed[:4000]
    train_x = train_pos + train_neg 
    val_x  = val_pos + val_neg

    train_y = [[1] for _ in train_pos] + [[0] for _ in train_neg]
    val_y  = [[1] for _ in val_pos] + [[0] for _ in val_neg]
    
    vocab = build_vocab(train_x)
    
    tweet = process_tweet(tweet)
    tweet = padded_sequence(tweet, vocab, max_len)
    tweet = np.array([tweet])

    prediction = model.predict(tweet, verbose=True)
    
    return prediction[0][0]


def infer():
    
    prediction = get_prediction_from_tweet(tweet, model, vocab, max_len)
    