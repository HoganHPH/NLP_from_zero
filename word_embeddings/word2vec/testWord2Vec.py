import random

from utils.treebank import StanfordTreebank


if __name__ == '__main__':
    random.seed(123)
    dataset = StanfordTreebank()
    print(dataset.getAllSentences())