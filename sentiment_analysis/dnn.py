import os
import subprocess

from pathlib import Path
from utils.load_yaml import yaml_load
import nltk

nltk.download("twitter_samples")
 
print("HELLO")

# root = Path(os.getcwd())
# data_file = 'data/TwitterSamples.yaml'
# data_file = os.path.join(root, data_file)
# print(data_file)


# data = yaml_load(data_file)
# script = data['download']
# print(data['download'])
# try:
#     result = subprocess.run(['bash', script], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     print("Shell script executed successfully.")
#     print(result.stdout.decode())
# except subprocess.CalledProcessError as e:
#     print("Error occurred while executing the shell script.")
#     print(e.stderr.decode())



