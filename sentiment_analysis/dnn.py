
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


def rnn_dnn_runner():
    
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

    print(f"There are {len(train_x)} sentences for training.")
    print(f"There are {len(train_y)} labels for training.\n")
    print(f"There are {len(val_x)} sentences for validation.")
    print(f"There are {len(val_y)} labels for validation.")

    vocab = build_vocabulary(train_x)
    num_words = len(vocab)

    print(f"Vocabulary contains {num_words} words\n")
