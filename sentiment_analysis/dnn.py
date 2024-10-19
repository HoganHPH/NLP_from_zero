
from utils.dataset_utils import setup_data, load_tweets


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

