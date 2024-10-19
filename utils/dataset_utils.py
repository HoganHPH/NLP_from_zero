import os
import subprocess

from pathlib import Path
from utils.load_yaml import yaml_load


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