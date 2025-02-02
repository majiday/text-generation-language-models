
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

#### GRU Model (rnn_gru.py)

Uses a GRU (Gated Recurrent Unit) layer for sequence processing. Includes an embedding layer, a GRU layer, and a dense output layer with softmax activation.

#### Bigram Model (bigram_model.py)

Uses a simple bigram approach for text generation. This model predicts the next word based on the previous word.

#### Trigram Model (trigram_model.py)

Uses a trigram approach for text generation. This model predicts the next word based on the previous two words.

## Running the Code

### Install Dependencies

Ensure you have the required libraries installed. You can use pip to install them:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Prepare Input Text

Place your text data in a file named `input.txt` in the same directory as the script. Ensure the text contains a start marker (e.g., "CHAPTER I") and an end marker (e.g., "End of Project Gutenberg's Alice's Adventures in Wonderland").

### Train the Model

Run the script corresponding to the model you want to train:

#### TransformerRNNModel

```bash
python transformer.py
```

#### GRU Model

```bash
python rnn_gru.py
```

#### Bigram Model

```bash
python bigram_model.py
```

#### Trigram Model

```bash
python trigram_model.py
```

The training process will print the training loss for each epoch.

### Generate Text

After training, the scripts will generate text starting from a given sequence. Adjust the `start_text`, `num_generate`, `temperature`, and `top_k` parameters as needed.

#### Example Usage for TransformerRNNModel

```python
start_text = "Alice was walking"
num_generate = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generated_text = generate_text(model, start_text, num_generate, token_to_id, id_to_token, vocab_size, device, temperature=0.8, top_k=40)
print(start_text + " " + generated_text)
```
#### sample output
```python
"""Alice was walking by his shoulder that it ’ll understand of them were lying on their tails in another moment on their tails in their tails down the Dodo away !
It ’s cook threw themselves flat , and washing_—extra , And they all over . Alice noticed , because she made of course , and began staring came skimming out of course ,
so many large replied . The Lobster Quadrille , when they were all sorts of course , in fact , that had come on which way , that savage when I ’ve seen her head downwards , 
and shoes"""
```

#### Example Usage for GRU Model

```python
start_string = "Alice was beginning"
print(generate_text_rnn(model, start_string))
```

#### sample output
```python
"""Alice was beginning saying giving nest dreadful knuckles chin howled shrieks shiver See myself denied remember dispute chance Sixteenth mine IX oblong Let But waistcoat
spirited modern pleased vinegar whom sides It roses IV last whatever lovely peeped doorway stupidest respectful face royal happened rabbit by growling deserved sort rapped W.
yours awfully serpents jurymen consented promised breath London Do Ugh except As silent butterfly rose themselves cheated III guilt tight VI seems unlocking Read course worry
banks hurt edge exclamation At writing If roof muttering were picture soup dainties London opposite Race shelves messages off tinkling Give teacups plenty birds Où seaside"""
```

#### Example Usage for Trigram Model

```python
start_words = "Alice was"
generated_text = generate_text(start_words, trigram_model, num_words=50)
print(generated_text)
```
#### sample output
```python
"""Alice was a sound of many footsteps , and the procession came opposite to Alice ; “ and most things
twinkled after that into a graceful zigzag , and the other end of half those long words , ” said
the March Hare had just begun “ Well ! I"""
```

#### Example Usage for Bigram Model

```python
generated_text = generate_text("Alice", bigram_model, num_words=50, threshold=0.01)
print(generated_text)
```

#### sample ouput
```python
"""Alice : “ And yesterday things between whiles . “ _ very glad to be lost something , it ’s all . The Gryphon .
“ I ca n’t see , and then treading on , ” cried the Dormouse shall sit down !
 ” cried Alice , but her"""
```
---

By following the above instructions, you can train and generate text using various models implemented in this repository.
