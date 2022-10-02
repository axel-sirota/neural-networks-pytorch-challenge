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
dataset1 = datasets.MNIST('./data', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('./data', train=False, download=True,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=256)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=256)

for tensor, target in train_loader:
    print(tensor.size())
    print(target.size())
    break

## 2. Create CNN and test it out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        nn.init.normal_(self.conv1.weight, mean=0, std=1.0)
        self.conv2 = nn.Conv2d(16, 16, 3, 1)
        nn.init.normal_(self.conv2.weight, mean=0, std=1.0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, 64)
        self.fc2 = nn.Linear(64, 10)

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
model.train()

print(summary(model))
for data, target in train_loader:
    output = model(data)
    print(output[0][:5])
    break

## 3. Train the network on MNIST dataset

optimizer = torch.optim.AdamW(model.parameters())
for epoch in range(1, EPOCHS + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

## 4. Evaluate model

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
