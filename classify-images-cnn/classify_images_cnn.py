## Initial imports and constants

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchvision import datasets, transforms

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
EPOCHS = 5

## 1. Prepare MNIST dataset and loaders

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Task 1: Build the train and test dataset and DataLoaders. For that you will need to:

# - Download the MNIST dataset using PyToch datasets API, transforming each image according to the transform function. Set these as the raw_train and raw_test sets
# We will do this because we will use Subset to only use a subset of images to make it run faster

# - Create dataloaders out of the test and train set with a batch size of 256 images

## INSERT TASK 1 CODE HERE

raw_train_set = None
raw_test_set = None
train_set = torch.utils.data.Subset(raw_train_set, range(0, len(raw_train_set), 5))
test_set = torch.utils.data.Subset(raw_test_set, range(0, len(raw_test_set), 5))
train_loader = None
test_loader = None

## END TASK 1 CODE

## Validation Task 1
## This is for validation only, after you finish the task feel free to remove the prints and the exit command

tensor, target = next(iter(train_loader))
print(tensor.size())
print(target.size())
exit(0)

## End of validation of task 1. (please remove prints and exits after ending it)


## 2. Create CNN and test it out

# Task 2: Create the layers of the Convolutional Neural Network we will use. For that consider that:

# - We want two conv layers with 16 3x3 kernels, each initialized as a gaussian
# - We want later two fully connected layers such that there are 64 hidden neurons in the last fc layer
# - Remember that we want to predict a number between 1 and 10

## INSERT TASK 2 CODE HERE

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = None  # FILLME
        self.conv2 = None  # FILLME
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = None  # FILLME
        self.fc2 = None  # FILLME

    ## END TASK 2 CODE

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)

## Validation Task 2
## This is for validation only, after you finish the task feel free to remove the prints and the exit command

print(summary(model))
data, target = next(iter(train_loader))
output = model(data)
print(output[0][:5])
exit(0)

## End of validation of task 2. (please remove prints and exits after ending it)


## 3. Train the network on MNIST dataset

# Task 3: Complete the training loop code.

## INSERT TASK 3 CODE HERE

model.train()
optimizer = torch.optim.AdamW(model.parameters())
for epoch in range(1, EPOCHS + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = None
        loss = None
        # Recall to execute the necessary methods such that the training loop works!
        if batch_idx % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


## END TASK 3 CODE

model.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
torch.save(model.state_dict(), "mnist_cnn.pt")
