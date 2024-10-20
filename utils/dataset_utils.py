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