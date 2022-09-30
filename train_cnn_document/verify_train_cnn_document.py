import pickle

import numpy as np
import pandas as pd
import torch
from textblob import TextBlob
from torch import nn
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

with open('weights.pkl', 'rb') as inp:
    weights = pickle.load(inp)

with open('maximum.pkl', 'rb') as inp:
    maximum = pickle.load(inp)

with open('word2vec_model.pkl', 'rb') as inp:
    word2vec_model = pickle.load(inp)


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
tokenizer = lambda x: TextBlob(x).words
model = NewsClassifier(embedding_size=50,
                       num_channels=50,
                       hidden_dim=100,
                       num_classes=4,
                       dropout_p=0.15,
                       pretrained_embeddings=weights
                      )

model.load_state_dict(torch.load('cnn_news_model_save.pth'))
model.eval()

review = np.array(["This place was fantastic", "I had such a bad time"])
X_val = torch.zeros(len(review), maximum).type(torch.LongTensor)
for index, text in pd.Series(review).items():
  ix = 0
  for word in tokenizer(text):
    if word not in word2vec_model:
      representation = 0
    else:
      representation = word2vec_model.index_to_key.index(word)
    X_val[index, ix] = representation
    ix += 1
prediction = model(X_val)
prediction.argmax(dim=1)
print('A-OK!')
