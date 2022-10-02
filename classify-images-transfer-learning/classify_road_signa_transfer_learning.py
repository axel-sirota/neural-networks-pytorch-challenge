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

# Task 1: Build the train and test dataset and DataLoaders. For that you will need to:

# - Load the Dataset from the images local folder
# - Create the DataLoaders with shuffle=True and a batch size of 5
# - Fetch all the classes from the all_data Dataset

## INSERT TASK 1 CODE HERE

data_dir = 'images'
all_data = None
train_set = torch.utils.data.Subset(all_data, [i for i in range(len(all_data)) if i % 5])
test_set = torch.utils.data.Subset(all_data, [i for i in range(len(all_data)) if not i % 5])

train_loader = None
test_loader = None
class_names = None

## END TASK 1 CODE


## Validation Task 1
## This is for validation only, after you finish the task feel free to remove the prints and the exit command

inputs, classes = next(iter(train_loader))
print(inputs.size())
print(classes)
exit(0)

## End of validation of task 1. (please remove prints and exits after ending it)

# 2. Setting the model

# Task 2: We will create the model, using as feature extractor ResNet 18, making the parameters non-trainable, and then adding a simple FC layer to our classes. For that you will need to:

# - Download the resnet18 weights from PyTorchHub
# - Setting the model parameters as non-trainable
# - Add a Linear FC layer mapping to our road sign classes
# - Set the appropriate loss (consider this model will not have a Softmax)
# - Finally set the optimizer as ab SGD such that only the last layer's parameters are changed on each step()

## INSERT TASK 2 CODE HERE

model = None
for param in model.parameters():
    ## FILLME
    pass
model = None

loss_function = None


optimizer_ft = None

## END TASK 2 CODE
## Validation Task 2
## This is for validation only, after you finish the task feel free to remove the prints and the exit command


summary(model)
print(model(inputs).size())
exit(0)

## End of validation of task 2. (please remove prints and exits after ending it)

# 3. Building the training routine

# Task 3: This time you will fill in the code to do the appropriate bookkeeping to know the running loss and accuracy. You will:

# - Calculate the running loss per batch as well as the amount on corrects
# - Calculate the Epoch loss an accuracy
# - Calculate the final test accuracy

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


        ## INSERT TASK 3 CODE HERE


        # statistics
        running_loss += None
        running_corrects += None
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(train_set), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    epoch_loss = None
    epoch_acc = None

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
    test_loss += None
    correct += None

## END TASK 3 CODE

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

assert 100. * correct / len(test_loader.dataset) > 60

torch.save(model.state_dict(), "road_signs_cnn.pt")
