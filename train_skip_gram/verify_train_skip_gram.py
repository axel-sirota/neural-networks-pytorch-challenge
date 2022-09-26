import pickle
import ast
from torch import nn
import pandas as pd
import torch


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


with open('word_to_ix.pkl', 'rb') as inp:
    word_to_ix = pickle.load(inp)

with open('vocab.pkl', 'rb') as inp:
    vocab = pickle.load(inp)

model = SkipGram(len(vocab), 50)
model.load_state_dict(torch.load('skip_gram_model_save.pth'))
model.eval()

test_set = pd.read_csv('test_data.csv', header=None)


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] if w in word_to_ix else 0 for w in context]
    return torch.tensor(idxs, dtype=torch.long)


for row in test_set.iterrows():
    _, context, target = row[1]
    target = ast.literal_eval(target)
    input = torch.tensor([word_to_ix[context]])

    log_probs = model(input)

    nn.functional.nll_loss(log_probs, make_context_vector(target, word_to_ix))

print('A-OK!')
