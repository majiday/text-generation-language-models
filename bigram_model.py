import random
from collections import defaultdict
import requests
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# URL of the text file for "Alice's Adventures in Wonderland"
url = 'https://www.gutenberg.org/files/11/11-0.txt'

# Fetch the text from the URL
response = requests.get(url)
response.encoding = 'utf-8-sig'  # Ensure correct encoding
text = response.text

# Clean the text
start_index = text.find("CHAPTER I")
end_index = text.find("End of Project Gutenberg's Alice's Adventures in Wonderland")
text = text[start_index:end_index]

# Use spaCy to tokenize the text
doc = nlp(text)

# Extract tokens
tokens = [token.text for token in doc if not token.is_space]

# Count the number of tokens
num_tokens = len(tokens)
print(f"Number of tokens: {num_tokens}")

# Generate bigrams
bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

# Create a dictionary to store the bigram probabilities
bigram_model = defaultdict(lambda: defaultdict(int))

# Populate the bigram model with counts
for word1, word2 in bigrams:
    bigram_model[word1][word2] += 1

# Convert counts to probabilities
for word1 in bigram_model:
    total_count = sum(bigram_model[word1].values())
    for word2 in bigram_model[word1]:
        bigram_model[word1][word2] /= total_count

# Function to generate text using the bigram model
def generate_text(start_word, model, num_words=50, threshold=0.01):
    current_word = start_word
    result = [current_word]
    for _ in range(num_words - 1):
        next_words = [word for word in model[current_word].keys() if model[current_word][word] >= threshold]
        next_word_probs = [model[current_word][word] for word in next_words]
        if not next_words:
            break  # Stop if there are no next words above the threshold
        next_word = random.choices(next_words, next_word_probs)[0]
        result.append(next_word)
        current_word = next_word
    return ' '.join(result)

# Generate text starting with "Alice"
generated_text = generate_text("Alice", bigram_model, num_words=50, threshold=0.01)
print(generated_text)

