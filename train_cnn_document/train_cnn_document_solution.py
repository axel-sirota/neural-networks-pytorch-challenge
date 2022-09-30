## Initial imports and constants
import pickle
import re
import warnings

import gensim
import nltk
import numpy as np
import pandas as pd
import torch
from textblob import TextBlob
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torchinfo import summary
from sklearn.model_selection import train_test_split

nltk.download('punkt')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

EMBEDDING_DIM = 50
EPOCHS = 250
BATCH_SIZE = 250
CORPUS_SIZE = 5000

## 1. Loading data and building vocabulary and tokenizer

path = './news.csv'
news = pd.read_csv(path, header=0)[:CORPUS_SIZE]


def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


news.title.apply(preprocess_text)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('emb_word2vec_format.txt')

news['label'] = news.category.map({'Business': 0, 'Sports': 1, 'Sci/Tech': 2, 'World':3})

weights = torch.FloatTensor(word2vec_model.vectors).to(device)
tokenizer = lambda x: TextBlob(x).words
vocab_size = len(word2vec_model.index_to_key)

def get_maximum_review_length(df):
  maximum = 0
  for ix, row in df.iterrows():
    candidate = len(tokenizer(row.title))
    if candidate > maximum:
      maximum = candidate
  return maximum

maximum = get_maximum_review_length(news)

X = torch.zeros(len(news), maximum).type(torch.LongTensor).to(device)
for index, row in news.iterrows():
  ix = 0
  for word in tokenizer(row.title):
    if word not in word2vec_model:
      representation = 0
    else:
      representation = word2vec_model.index_to_key.index(word)
    X[index, ix] = representation
    ix += 1
y = news.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
y_train = torch.Tensor(y_train.values).type(torch.LongTensor).to(device)
y_test = torch.Tensor(y_test.values).type(torch.LongTensor).to(device)


class NewsClassifier(nn.Module):
    def __init__(self, embedding_size, num_channels,
                 hidden_dim, num_classes, dropout_p,
                 pretrained_embeddings):
        """
        Args:
            embedding_size (int): size of the embedding vectors
            num_channels (int): number of convolutional kernels per layer
            hidden_dim (int): the size of the hidden dimension
            num_classes (int): the number of classes in classification
            dropout_p (float): a dropout parameter
            pretrained_embeddings (torch.Tensor): Weights of pretraine embedding
        """
        super(NewsClassifier, self).__init__()
        self.emb = nn.Embedding.from_pretrained(weights)
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU()
        )
        self._dropout_p = dropout_p
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """

        # embed and permute so features are channels
        x_embedded = self.emb(x_in).permute(0, 2, 1)
        features = self.convnet(x_embedded)

        # average and remove the extra dimension
        remaining_size = features.size(dim=2)
        features = nn.functional.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = nn.functional.dropout(features, p=self._dropout_p)

        # mlp classifier
        intermediate_vector = nn.functional.relu(nn.functional.dropout(self.fc1(features), p=self._dropout_p))
        prediction_vector = self.fc2(intermediate_vector)
        prediction_vector = nn.LogSoftmax(dim=-1)(prediction_vector)

        return prediction_vector

model = NewsClassifier(embedding_size=EMBEDDING_DIM,
                       num_channels=50,
                       hidden_dim=100,
                       num_classes=4,
                       dropout_p=0.15,
                       pretrained_embeddings=weights
                      ).to(device)
summary(model)

def loss(y_pred, y):
  return nn.functional.nll_loss(y_pred, y)

def metric(y_pred, y):  # -> accuracy
  return (1 / len(y)) * ((y_pred.argmax(dim = 1) == y).sum())

optimizer = torch.optim.AdamW(model.parameters())

model.train()
for i in range(EPOCHS):
  y_pred = model(X_train)
  xe = loss(y_pred, y_train)
  accuracy = metric(y_pred, y_train)
  xe.backward()
  if i % 50 == 0:
    print("Loss: ", xe, " Accuracy ", accuracy.data.item())
  optimizer.step()
  optimizer.zero_grad()

model.eval()
y_test_pred = model(X_test)
print(f'Model accuracy is {metric(y_test_pred, y_test)}')

torch.save(model.state_dict(), 'cnn_news_model_save.pth')

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
#
# # sample usage
save_object(weights, 'weights.pkl')
save_object(maximum, 'maximum.pkl')
save_object(word2vec_model, 'word2vec_model.pkl')
