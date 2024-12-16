import random

# 20 unique words
dog_cat_words = ['dog', 'cat', 'pet', 'house', 'animal', 'sleep', 'play']
family_words = ['girl', 'boy', 'father', 'mother', 'family', 'house', 'marriage']
king_queen_words = ['crown', 'queen', 'king', 'empire', 'country', 'rule', 'castle']


# Shuffle these words and generate random long sequences
dog_cat_text = ''
family_text = ''
king_queen_text = ''

for _ in range(10_000):
    random.shuffle(dog_cat_words)
    dog_cat_text = dog_cat_text + ' ' + ' '.join(dog_cat_words)
    random.shuffle(family_words)
    family_text = family_text + ' ' + ' '.join(family_words)
    random.shuffle(king_queen_words)
    king_queen_text = king_queen_text + ' ' + ' '.join(king_queen_words)

small_corpus = dog_cat_text + ' ' + family_text + ' ' + king_queen_text

# Extract the dataset
file_name = "small_corpus.txt"
with open(file_name, 'w') as file:
    file.write(small_corpus) 