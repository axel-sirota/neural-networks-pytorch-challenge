import copy
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet18_Weights

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
EPOCHS = 10

# 1. Preparing the Dataset and Dataloader

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'images'
all_data = datasets.ImageFolder(os.path.join(data_dir), transform=transform)
train_set = torch.utils.data.Subset(all_data, [i for i in range(len(all_data)) if i % 5])
test_set = torch.utils.data.Subset(all_data, [i for i in range(len(all_data)) if not i % 5])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=5, shuffle=True, drop_last=True)
class_names = train_set.dataset.classes

# Get a batch of training data
inputs, classes = next(iter(train_loader))
print(inputs.size())
print(classes)

# 2. Setting the model

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_in_features = model.fc.in_features
model.fc = nn.Linear(num_in_features, len(classes))
model = model.to(device)

loss_function = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

summary(model)

# 3. Building the training  routine

best_acc = 0
model.train()  # Set model to training mode
for epoch in range(EPOCHS):
    print(f'Epoch {epoch}/{EPOCHS - 1}\n\n')
    print('-' * 10)
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer_ft.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(train_set), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    epoch_loss = running_loss / len(train_set)
    epoch_acc = running_corrects.double() / len(train_set)

    print(f'\nLoss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    # deep copy the model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

print(f'Best Train Acc: {best_acc:4f}')

model.eval()
test_loss = 0
correct = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    loss = loss_function(outputs, labels)
    test_loss += loss.item() * inputs.size(0)
    correct += torch.sum(preds == labels.data)

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

assert 100. * correct / len(test_loader.dataset) > 60

torch.save(model.state_dict(), "road_signs_cnn.pt")
