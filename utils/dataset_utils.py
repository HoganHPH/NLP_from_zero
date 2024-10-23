import os
import re
import string
import subprocess

from pathlib import Path
from utils.load_yaml import yaml_load

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet 
from nltk.stem import WordNetLemmatizer

def setup_data():
    root = Path(os.getcwd())
    data_file = 'data/TwitterSamples.yaml'
    data_file = os.path.join(root, data_file)

    data = yaml_load(data_file)
    script = data['download_file']
    download_file = os.path.join(os.getcwd(), script)

    try:
        
        result = subprocess.run(['bash', download_file], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Shell script executed successfully.")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing the shell script.")
        print(e.stderr.decode())
        
def load_tweets():
    from nltk.corpus import twitter_samples
    
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')  
    return all_positive_tweets, all_negative_tweets


def pos_tag_convert(nltk_tag: str) -> str:
    '''Converts nltk tags to tags that are understandable by the lemmatizer.
    
    Args:
        nltk_tag (str): nltk tag
        
    Returns:
        _ (str): converted tag
    '''
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return wordnet.NOUN
    

def process_tweet(tweet):
    
    '''
    Input: 
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    
    '''
    """
    pre-processing procedure:
    - tokenizing the sentence (splitting to words)
    - removing stock market tickers like $GE
    - removing old style retweet text "RT"
    - removing hyperlinks
    - removing hashtags
    - lowercasing
    - removing stopwords and punctuation
    - stemming
    """
    stopwords_english = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = nltk.pos_tag(tokenizer.tokenize(tweet))

    tweets_clean = []
    for word in tweet_tokens:
        if (word[0] not in stopwords_english and # remove stopwords
            word[0] not in string.punctuation): # remove punctuation
            stem_word = lemmatizer.lemmatize(word[0], pos_tag_convert(word[1]))
            tweets_clean.append(stem_word)
    return tweets_clean


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
    padded_tensor = np.array(padded_tensor)
    return padded_tensor


def get_traintest_tweet():
    
    setup_data()

    print("Loading train test tweet dataset")
    all_positive_tweets, all_negative_tweets = load_tweets()
    
    # Process all the tweets: tokenize the string, remove tickers, handles, punctuation and stopwords, stem the words
    all_positive_tweets_processed = [process_tweet(tweet) for tweet in all_positive_tweets]
    all_negative_tweets_processed = [process_tweet(tweet) for tweet in all_negative_tweets]
    
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

    # Assign each unique word in corpus to an unique integer
    vocab = build_vocab(train_x)
    num_words = len(vocab)

    # Find the largest value of the length of every sentences (or max length)
    max_len = max_length(train_x, val_x)
    
    # Pad the sequence to make sure they are in the same length
    train_x_padded = [padded_sequence(tweet, vocab, max_len) for tweet in train_x]
    val_x_padded = [padded_sequence(tweet, vocab, max_len) for tweet in val_x]
    
    print("\nSuccess!")
    print(f"There are {len(train_x_padded)} sequences in train dataset")
    print(f"There are {len(val_x_padded)} sequences in test dataset")
    
    return (train_x_padded, val_x_padded, train_y, val_y)