import copy
import os
import warnings

import torch
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchinfo import summary
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet18_Weights

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
EPOCHS = 25

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


model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_in_features = model.fc.in_features
model.fc = nn.Linear(num_in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()

summary(model)
# print(inputs.size())
# print(model(inputs))
# print(F.softmax(model(inputs), dim=1))
# print(torch.max(F.softmax(model(inputs), dim=1), dim=1))
# print(classes[torch.max(F.softmax(model(inputs), dim=1), dim=1)[1]])
# print(classes[torch.max(F.softmax(model(inputs), dim=1), dim=1)[1]] == classes.data)
# # print(torch.max(model(inputs), 1)[0].size())
# # print(criterion(model(inputs), classes))
# # print(torch.max(model(inputs), 1)[1] == classes.data)

# exit(0)

optimizer_ft = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

best_acc = 0
model.train()  # Set model to training mode
for epoch in range(EPOCHS):
    print(f'Epoch {epoch}/{EPOCHS - 1}')
    print('-' * 10)
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        # forward
        # track history if only in train
        with torch.enable_grad():
            predicted_probs = F.softmax(model(inputs), dim=1)
            class_prediction = classes[torch.max(F.softmax(model(inputs), dim=1), dim=1)[1]]
            loss = criterion(predicted_probs, labels)
            loss.backward()
            optimizer_ft.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(class_prediction == labels.data)
    epoch_loss = running_loss / len(train_set)
    epoch_acc = running_corrects.double() / len(train_set)

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # deep copy the model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

print(f'Best val Acc: {best_acc:4f}')

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
