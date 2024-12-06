import random

from utils.treebank import StanfordTreebank


if __name__ == '__main__':
    random.seed(123)
    dataset = StanfordTreebank()
    sentences = dataset.getRawSentences()
    # s = sentences * 30
    # print(len(s))
    # print(len(sentences))
    sentence_label = dataset.getSentLabels()
    print(sentence_label[0])
    # tokens = dataset.tokenize()
    # print(sentences[0])
    # print(s[0])
    # s = dataset.getAllSentences()
    # print(s[1])
    # print()