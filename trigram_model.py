import random
from collections import defaultdict
import requests
import spacy

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

# Generate trigrams
trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]

# Create a dictionary to store the trigram probabilities
trigram_model = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# Populate the trigram model with counts
for word1, word2, word3 in trigrams:
    trigram_model[word1][word2][word3] += 1

# Convert counts to probabilities
for word1 in trigram_model:
    for word2 in trigram_model[word1]:
        total_count = sum(trigram_model[word1][word2].values())
        for word3 in trigram_model[word1][word2]:
            trigram_model[word1][word2][word3] /= total_count

# Function to generate text using the trigram model
def generate_text(start_words, model, num_words=50, threshold=0.01):
    words = start_words.split()
    result = words[:]
    current_word, last_word = words[-2], words[-1]
    for _ in range(num_words - 2):
        next_words = [word for word in model[current_word][last_word].keys() if model[current_word][last_word][word] >= threshold]
        next_word_probs = [model[current_word][last_word][word] for word in next_words]
        if not next_words:
            break  # Stop if there are no next words above the threshold
        next_word = random.choices(next_words, next_word_probs)[0]
        result.append(next_word)
        current_word, last_word = last_word, next_word
    return ' '.join(result)


# Example usage
start_words = "Alice was"
generated_text = generate_text(start_words, trigram_model, num_words=50)

#print(generated_text)
words = generated_text.split()
# Print 10 words per line.
for i in range(0, len(words), 20):
    print(" ".join(words[i:i+20]))