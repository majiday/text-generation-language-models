# Text Generation using Various Models

This repository contains scripts that implement text generation models using different architectures including Transformer with RNN, GRU, Bigram, and Trigram models. These models are trained on a given text corpus and can generate text based on a provided start sequence.

## Key Components

### Common Components

1. **Imports and Dependencies**:
   - The scripts use libraries like `spaCy` for tokenization, `numpy` for array manipulations, and `torch` or `tensorflow` for building and training the models.

2. **Data Preparation**:
   - The text data is loaded from `input.txt`, cleaned, and tokenized using `spaCy`.
   - Sequences of tokens and their corresponding next words are created and converted to integer IDs.

### Model Architectures

#### TransformerRNNModel (transformer.py)

Combines a Transformer encoder with an RNN to process token sequences and predict the next token. Includes an embedding layer, a Transformer encoder layer, an RNN layer, and a fully connected layer.

#### GRU Model (run_gru.py)

Uses a GRU (Gated Recurrent Unit) layer for sequence processing. Includes an embedding layer, a GRU layer, and a dense output layer with softmax activation.

#### Bigram Model (bigram_model.py)

Uses a simple bigram approach for text generation. This model predicts the next word based on the previous word.

#### Trigram Model (trigram_model.py)

Uses a trigram approach for text generation. This model predicts the next word based on the previous two words.

## Running the Code

### Install Dependencies

Ensure you have the required libraries installed. You can use pip to install them:

```bash
pip install spacy torch numpy requests tensorflow
python -m spacy download en_core_web_sm
