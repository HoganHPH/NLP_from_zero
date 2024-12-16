import json
import matplotlib.pyplot as plt

import torch

from preprocess_data import extract_unique_words
from cbow_model import NaiveCbow


def cosine_similarity(v1, v2):
    return (v1 @ v2) / (torch.norm(v1) * torch.norm(v2))

def most_similar(word, word_dict, top_k=5):
    if word not in word_dict:
        raise ValueError(f"{word} not found in the word dictionary.")

    query_vector = word_dict[word]

    # Calculate cosine similarity with all other words in the dictionary
    similarities = {}
    for other_word, other_vector in word_dict.items():
        if other_word != word:
            similarity = cosine_similarity(query_vector, other_vector)
            similarities[other_word] = similarity

    # Sort the words by similarity in descending order
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Get the top-k most similar words
    top_similar_words = sorted_similarities[:top_k]

    return top_similar_words


if __name__ == "__main__":
    
    file_path = 'small_corpus.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    unique_words = extract_unique_words(text)
    
    # Define model and load pretrained
    VOCAB_SIZE = len(unique_words)
    VECTOR_DIM = 2
    model = NaiveCbow(VOCAB_SIZE, VECTOR_DIM)

    model.load_state_dict(torch.load("cbow.pt", weights_only=True))
    model.eval()

    # Word Vectors
    params = list(model.parameters())
    word_vectors = params[0].detach()

    # Create a dictionary with the same order mapping
    word_dict = {word: vector for word, vector in zip(unique_words, word_vectors)}

    print("\n====> Word dict:")
    print(word_dict)

    x_coords, y_coords = zip(*[word_dict[word].numpy() for word in list(word_dict.keys())])

    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, marker='o', color='blue')

    for i, word in enumerate(list(word_dict.keys())):
        plt.annotate(word, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.title('Word Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.savefig("embedding_space.jpg")
    plt.show()

    
    # Specify the file path where you want to save the JSON file
    file_path = 'pretrained_word_vectors.json'

    # Convert torch.Tensor objects to lists
    word_vec_for_export = word_dict.copy()
    for key, value in word_vec_for_export.items():
        if isinstance(value, torch.Tensor):
            word_vec_for_export[key] = value.tolist()

    # Use json.dump to write the modified dictionary to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(word_vec_for_export, json_file, indent=2)

    print(f'Dictionary has been exported to {file_path}')