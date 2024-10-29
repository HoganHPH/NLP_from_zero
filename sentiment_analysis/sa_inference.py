import numpy as np
from utils.dataset_utils import load_tweets, build_vocab, max_length, process_tweet, padded_sequence


def get_sentiment_from_tweet(tweet, model):
    
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
    max_len = max_length(train_x, val_x)
    
    tweet = process_tweet(tweet)
    tweet = padded_sequence(tweet, vocab, max_len)
    tweet = np.array([tweet])

    prediction = model.predict(tweet, verbose=True)
    
    return prediction[0][0]