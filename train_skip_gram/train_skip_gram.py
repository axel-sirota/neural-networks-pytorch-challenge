## Initial imports and constants
import pickle
import warnings
import nltk
import numpy as np
import pandas as pd
import torch
from textblob import TextBlob
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torchinfo import summary

nltk.download('punkt')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

EMBEDDING_DIM = 50
EPOCHS = 2
BATCH_SIZE = 250
train_size = 5000
CORPUS_SIZE = 1000

## 1. Loading data and building vocabulary and tokenizer

path = './yelp_reviews.csv'
yelp = pd.read_csv(path)
text_df = yelp.text
text_sampled = text_df.sample(CORPUS_SIZE, random_state=42).values

# Task 1: Build the vocabulary and the tokenizer:


## INSERT TASK 1 CODE HERE

tokenizer = None
vocab = None

## END TASK 1 CODE

## Validation Task 1
## This is for validation only, after you finish the task feel free to remove the prints and the exit command


print(f'Vocab size is {len(vocab)}')
print(vocab(tokenizer("This is a fantastic ice cream")))
exit(0)

## End of validation of task 1. (please remove prints and exits after ending it)

## 2. Building dataset with context

vocab_size = len(vocab)
word_to_ix = {}
for sentence in text_sampled:
    for word in tokenizer(sentence):
        word_to_ix[word] = vocab([word])[0]
ix_to_word = {ix: word for word, ix in word_to_ix.items()}

# Task 2:
#
# - Create the dataset in the format Skip Gram expects, still as words
# - Create the function that gets a lists of words and returns a tensor of their ix with respect to word_to_ix mapping


original_data = []
for sentence in text_sampled:
    ## INSERT TASK 2 CODE HERE
    tokenized_sentence = None
    None

data = original_data[:train_size]


def make_context_vector(context, word_to_ix):
    # FILLME
    pass


## END TASK 2 CODE

## Validation Task 2
## This is for validation only, after you finish the task feel free to remove the prints and the exit command

print(data[0])
print(make_context_vector(['sausage', 'hotdog', 'burger', 'pasta'], word_to_ix=word_to_ix))
exit(0)


## End of validation of task 2. (please remove prints and exits after ending it)

## 3. Build Model and validate it

## Task 3: Create the SkipGram Model, remember it should have an Embedding layer and a linear layer for each of the 4 predicted words

## INSERT TASK 3 CODE HERE

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.linear1 = None
        self.activation_function2 = None

    def forward(self, inputs):
        # FILLME
        pass


## END TASK 3 CODE

model = SkipGram(vocab_size, EMBEDDING_DIM).to(device)


def loss_function(y_pred, y):
    return nn.functional.nll_loss(y_pred, y)


optimizer = torch.optim.AdamW(model.parameters())


def validate_model(context, target):
    input = torch.tensor([word_to_ix[context]]).to(device)
    log_probs = model(input)
    loss_function(log_probs, make_context_vector(target, word_to_ix))
    return [ix_to_word[ix] for ix in torch.argmax(log_probs, dim=-1).detach().numpy()]


## Validation Task 3
## This is for validation only, after you finish the task feel free to remove the prints and the exit command


context, target = next(iter(data))
predictions = validate_model(context, target)  # This shouldn't fail
print(f'Model summary: {summary(model)}')
print(f'Input, true value: {context, target}\n')
print(f'Prediction: {predictions}')
exit(0)

## End of validation of task 3. (please remove prints and exits after ending it)


# 4. Training routine. Please run the file entirely.

for epoch in range(EPOCHS):
    total_loss = 0
    n_rows = 1
    batches = 1
    for context, target in data:
        input_word_tensor = torch.tensor([word_to_ix[context]]).to(device)
        context_pred_tensor = make_context_vector(target, word_to_ix)
        log_probs = model(input_word_tensor)
        total_loss += loss_function(log_probs, context_pred_tensor)
        if n_rows > BATCH_SIZE:
            print(f"-" * 59)
            print(f"Epoch: {epoch}, Batch: {batches}, Loss: {total_loss}")
            batches += 1
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = 0
            n_rows = 0
        n_rows += 1

torch.save(model.state_dict(), 'skip_gram_model_save.pth')


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


# sample usage
save_object(word_to_ix, 'word_to_ix.pkl')
save_object(vocab, 'vocab.pkl')
pd.DataFrame(original_data[train_size:train_size + 200]).to_csv('test_data.csv', header=False)
