import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from preprocess_data import extract_unique_words, generate_cbows, one_hot_encoding
from datasets import CustomDataset
from cbow_model import NaiveCbow


def train_model(model, train_dataloader, validation_dataloader, epochs, learning_rate, verbose=False):

    # Create the loss function
    loss_fn = nn.CrossEntropyLoss()
    # Create the optimizer object
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Log the loss values
    train_set_loss_log = []
    validation_set_loss_log = []

    for epoch in range(epochs):
        if verbose: print("Epoch: ", epoch + 1)
        # Training mode on
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for inputs_batch, outputs_batch in train_dataloader:

            y_train_logits = model(inputs_batch)
            train_loss = loss_fn(y_train_logits, outputs_batch)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()
            num_train_batches += 1            

        # Calculate average training loss for the epoch
        average_train_loss = total_train_loss / num_train_batches
        train_set_loss_log.append(average_train_loss)
        
        # Eval mode on
        model.eval()            
        total_validation_loss = 0.0
        num_validation_batches = 0

        with torch.inference_mode():
            for inputs_batch, outputs_batch in validation_dataloader:
                # Evaluate the validation loss
                y_val_logits = model(inputs_batch)
                validation_loss = loss_fn(y_val_logits, outputs_batch)

                total_validation_loss += validation_loss.item()
                num_validation_batches += 1
        
        # Calculate average validation loss for the epoch
        average_validation_loss = total_validation_loss / num_validation_batches
        validation_set_loss_log.append(average_validation_loss) 

        if verbose: print("Train Loss: ", average_train_loss, "|||", "Validation Loss: ", average_validation_loss)
    torch.save(model.state_dict(), "cbow.pt")
    return model, train_set_loss_log, validation_set_loss_log


if __name__ == "__main__":
    file_path = 'small_corpus.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    unique_words = extract_unique_words(text)

    # Create cbows
    cbows = generate_cbows(text, window_size=3)
    print("Length of cbows : ", len(cbows))

    # Display the results
    for context_words, target_word in cbows[:3]:
        print(f'Context Words: {context_words}, Target Word: {target_word}')

    # Create one-hot encodings for each word
    one_hot_encodings = {word: one_hot_encoding(word, unique_words) for word in unique_words}
    print("One hot vector of word 'king' : ", one_hot_encodings['king'])

    # Convert CBOW pairs to vector pairs
    cbow_vector_pairs = [([one_hot_encodings[word] for word in context_words], one_hot_encodings[target_word]) for context_words, target_word in cbows]

    # Sum the context vectors to get a single context vector
    cbow_vector_pairs = [(torch.sum(torch.stack(context_vectors), dim=0), target_vector) for context_vectors, target_vector in cbow_vector_pairs]

    print("\nContext vector : ", cbow_vector_pairs[0][0])
    print("\nTarget vector : ", cbow_vector_pairs[0][1])


    # Shuffle pairs before training
    cbow_vector_pairs = random.sample(cbow_vector_pairs, len(cbow_vector_pairs))

    # Train and Val split
    split_index = int(len(cbow_vector_pairs) * 0.90)

    # Split the data into training and test sets
    train_dataset = CustomDataset(cbow_vector_pairs[:split_index])
    test_dataset = CustomDataset(cbow_vector_pairs[split_index:])

    # Set batch size
    batch_size = 64  # You can adjust this based on your requirements

    # Create DataLoader for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nSize of train dataloader : ", len(train_dataloader))
    print("Size of val dataloader : ", len(validation_dataloader))

    # Define model
    VOCAB_SIZE = len(unique_words)
    VECTOR_DIM = 2
    model = NaiveCbow(VOCAB_SIZE, VECTOR_DIM)
    
    model, train_set_loss_log, validation_set_loss_log = train_model(model, train_dataloader, validation_dataloader, 
                                                                 epochs=10, learning_rate=0.001, verbose=True)
    
    plt.plot(train_set_loss_log, color='red', label='train_loss')
    plt.plot(validation_set_loss_log, color='blue', label='validation_loss')

    plt.title("Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.legend()
    plt.savefig("training_log.jpg")
    plt.show()
    print("\nTRAINING SUCCESSFULLY!")