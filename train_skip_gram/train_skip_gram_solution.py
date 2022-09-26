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
EPOCHS = 10
BATCH_SIZE = 250
train_size = 5000
CORPUS_SIZE = 1000

## 1. Loading data and building vocabulary and tokenizer

path = './yelp_reviews.csv'
yelp = pd.read_csv(path)
text_df = yelp.text
text_sampled = text_df.sample(CORPUS_SIZE, random_state=42).values


def build_vocab(data_iter, text_tokenizer):
    """Builds vocabulary from iterator"""
    vocab = build_vocab_from_iterator(
        yield_tokens(data_iter, text_tokenizer),
        specials=["<unk>"],
        min_freq=10,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)


tokenizer = lambda x: TextBlob(x).words
vocab = build_vocab(text_sampled, tokenizer)
print(f'Vocab size is {len(vocab)}')
print(vocab(tokenizer("This is a fantastic ice cream")))

## 2. Building dataset with context

vocab_size = len(vocab)
word_to_ix = {}
for sentence in text_sampled:
    for word in tokenizer(sentence):
        word_to_ix[word] = vocab([word])[0]
ix_to_word = {ix: word for word, ix in word_to_ix.items()}

original_data = []
for sentence in text_sampled:
    tokenized_sentence = tokenizer(sentence)
    for i in range(2, len(tokenized_sentence) - 2):
        target = [tokenized_sentence[i - 2], tokenized_sentence[i - 1],
                  tokenized_sentence[i + 1], tokenized_sentence[i + 2]]
        context = tokenized_sentence[i]
        original_data.append((context, target))
data = original_data[:train_size]


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] if w in word_to_ix else 0 for w in context]
    return torch.tensor(idxs, dtype=torch.long).to(device)


print(data[0])
print(make_context_vector(['sausage', 'hotdog', 'burger', 'pasta'], word_to_ix=word_to_ix))


## 3. Build Model and validate it


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 4 * vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear1(embeds).reshape(4, -1)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word).view(1, -1)


model = SkipGram(vocab_size, EMBEDDING_DIM).to(device)


def loss_function(y_pred, y):
    return nn.functional.nll_loss(y_pred, y)


optimizer = torch.optim.AdamW(model.parameters())

context, target = next(iter(data))

input = torch.tensor([word_to_ix[context]]).to(device)

log_probs = model(input)

loss_function(log_probs, make_context_vector(target, word_to_ix))

print(f'Model summary: {summary(model)}')
print(f'Input, true value: {context, target}\n')
print(f'Prediction: {[ix_to_word[ix] for ix in torch.argmax(log_probs, dim=-1).detach().numpy()]}')
# 4. Training routine

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

# Final validation

context, target = next(iter(data))

input = torch.tensor([word_to_ix[context]]).to(device)

log_probs = model(input)

loss_function(log_probs, make_context_vector(target, word_to_ix))

print(f'Target, true value: {context, target}\n')
print(f'Prediction: {[ix_to_word[ix] for ix in torch.argmax(log_probs, dim=-1).detach().numpy()]}')

torch.save(model.state_dict(), 'skip_gram_model_save.pth')


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


# sample usage
save_object(word_to_ix, 'word_to_ix.pkl')
save_object(vocab, 'vocab.pkl')
pd.DataFrame(original_data[train_size:train_size+200]).to_csv('test_data.csv', header=False)
