## Initial imports and constants
import pickle
import re
import warnings

import gensim
import nltk
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from torch import nn
from torchinfo import summary

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

def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


path = './news.csv'

# Task 1: Build the DataFrame to use.
#
# - Load the news.csv file into a pandas dataframe
# - Apply the preprocessing function to the title column
# - Map the categories into numbered labels


## INSERT TASK 1 CODE HERE

news = None

## END TASK 1 CODE

## Validation Task 1
## This is for validation only, after you finish the task feel free to remove the prints and the exit command


print(f'News DF columns are is {news.columns}')
print(f'Is ? in the first title? : {"?" in news.iloc[0]["title"] == False}')
exit(0)

## End of validation of task 1. (please remove prints and exits after ending it)

## 2. Building dataset with help of Vocabulary model

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('emb_word2vec_format.txt')
weights = torch.FloatTensor(word2vec_model.vectors).to(device)
tokenizer = lambda x: TextBlob(x).words
vocab_size = len(word2vec_model.index_to_key)

# Task 2:
#
# - Implement get_maximum_review_length that takes the news DF and returns the maximum length in words of any sentence in the title column
# - Create the X dataset iterating over the news dataset.
#    * For every wor and every sentence in the title if the word is in the vocabulary, set the value for that word as the index
#    * If it is not in the model, set a tensor of 0

# For example, if the vocab is {cat, dog} and my df is ["My cat", "Dog is good"]
# Then X will be: [[0,1,0], [2,0,0]]. Be sure to understand why!
#

## INSERT TASK 2 CODE HERE

def get_maximum_review_length(df):
    """ Figures out how long should the tensors be to accomodate all reviews"""
    maximum = 0
    # FILLME
    return maximum

maximum = get_maximum_review_length(news)

X = torch.zeros(len(news), maximum).type(torch.LongTensor).to(device)
# FILLME


## END TASK 2 CODE

## Validation Task 2
## This is for validation only, after you finish the task feel free to remove the prints and the exit command

print(maximum)
print(X[0])
exit(0)


## End of validation of task 2. (please remove prints and exits after ending it)

## 3. Build Model and validate it

y = news.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = torch.Tensor(y_train.values).type(torch.LongTensor).to(device)
y_test = torch.Tensor(y_test.values).type(torch.LongTensor).to(device)

## Task 3: Create the News Classifier Model. It should have:

# - An Embedding layer from the pretrained weights
# - A sequence of two sets of Conv1d layer and ELU layer with a kernel size of 3
# - A fully connected layer with a hidden dimension
# - A final fully connected layer

# Part of the task is figuring out the correct dimensions.

## INSERT TASK 3 CODE HERE

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
        self.emb = None
        self.convnet = nn.Sequential(None)
        self._dropout_p = dropout_p
        self.fc1 = None
        self.fc2 = None

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

## END TASK 3 CODE

model = NewsClassifier(embedding_size=EMBEDDING_DIM,
                       num_channels=50,
                       hidden_dim=100,
                       num_classes=4,
                       dropout_p=0.15,
                       pretrained_embeddings=weights
                       ).to(device)

## Validation Task 3
## This is for validation only, after you finish the task feel free to remove the prints and the exit command

print(f'Model summary: {summary(model)}')
print(f'Prediction: {model(X[0])}')
exit(0)

## End of validation of task 3. (please remove prints and exits after ending it)

def loss(y_pred, y):
    return nn.functional.nll_loss(y_pred, y)


def metric(y_pred, y):  # -> accuracy
    return (1 / len(y)) * ((y_pred.argmax(dim=1) == y).sum())


optimizer = torch.optim.AdamW(model.parameters())

# 4. Training routine. Please run the file entirely.

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
