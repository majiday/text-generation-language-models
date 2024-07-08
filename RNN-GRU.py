import random
from collections import defaultdict
import requests
import spacy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Path to the text-file
file_path = 'input.txt'

# Open the text-file and read the content
with open(file_path, 'r', encoding='utf-8-sig') as file:
    text = file.read()

# Clean the text
start_index = text.find("CHAPTER I")
end_index = text.find("End of Project Gutenberg's Alice's Adventures in Wonderland")
text = text[start_index:end_index]

# Use spaCy to tokenize the text
doc = nlp(text)

# Extract tokens
tokens = [token.text for token in doc if not token.is_space]



#####
# Define sequence length
seq_length = 50

# Generate sequences of tokens
sequences = []
next_words = []

for i in range(len(tokens) - seq_length):
    sequences.append(tokens[i:i + seq_length])
    next_words.append(tokens[i + seq_length])

# Convert sequences and next words to integers
unique_tokens = list(set(tokens))
token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

sequences = np.array([[token_to_id[token] for token in seq] for seq in sequences])
next_words = np.array([token_to_id[token] for token in next_words])



# model 
# Define model parameters
vocab_size = len(unique_tokens)
embedding_dim = 100
rnn_units = 128




model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=seq_length),
    GRU(rnn_units, return_sequences=False),
    #Dropout(0.5),
    Dense(vocab_size, activation='softmax')
])

learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
model.summary()


# train the model 
# Define training parameters
batch_size = 128
epochs = 20

model.fit(
    sequences,
    next_words,
    batch_size=batch_size,
    epochs=epochs
)



# Function to generate text using the RNN model
def generate_text_rnn(model, start_string, num_generate=100):
    # Convert start string to tokens
    input_eval = [token_to_id[s] for s in start_string.split()]

    # Pad input to match the sequence length expected by the model
    if len(input_eval) < seq_length:
        input_eval = [0] * (seq_length - len(input_eval)) + input_eval
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0

    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)  # Remove batch dimension

        # Ensure the predictions are in the right shape for tf.random.categorical
        predictions = tf.expand_dims(predictions, 0)  # Add batch dimension back

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims(input_eval.numpy().tolist()[0][1:] + [predicted_id], 0)
        text_generated.append(id_to_token[predicted_id])

    return start_string + ' ' + ' '.join(text_generated)


# Generate text
start_string = "Alice was beginning"
print(generate_text_rnn(model, start_string))