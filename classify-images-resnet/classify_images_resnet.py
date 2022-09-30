## Initial imports and constants
import warnings
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchinfo import summary


warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

## 1. Load images and preprocess them.

image_batch = [Image.open('bear.png').convert('RGB'), Image.open('condor.png').convert('RGB')]

# Task 1: Build the dataset. For that you need to apply to each image:

# - Resize each value to 256
# - Make the images 224 x 224 (it already has 3 channels due to being RGB )
# - Normalize it
#
# And finally stack all the tensors

## INSERT TASK 1 CODE HERE

preprocess = transforms.Compose([
    None,  # FILLME
    None,  # FILLME
    None,  # FILLME
    None,  # FILLME
])

input_tensor = None
input_tensor.to(device)

## END TASK 1 CODE
print(input_tensor.size())

## Validation Task 1
## This is for validation only, after you finish the task feel free to remove the prints and the exit command


print(input_tensor.size())
exit(0)

## End of validation of task 1. (please remove prints and exits after ending it)

## 2. Load the model and calculate the probabilities vector

# Task 2: Use PyTorch Hub to load a pretrained resnet18 model and with it calculate the probabilities vector on both images

## INSERT TASK 2 CODE HERE

model = None
model.eval()
output = model(input_tensor)

# Tensors of shape 1000, with probability scores over Imagenet's 1000 classes
probabilities_bear = None
probabilities_condor = None

## END TASK 2 CODE

## Validation Task 2
## This is for validation only, after you finish the task feel free to remove the prints and the exit command

print(probabilities_bear[:5])
print(probabilities_condor[:5])
print(output[0][:5])
print(summary(model))
exit(0)

## End of validation of task 2. (please remove prints and exits after ending it)

# 3. Get the top 5 classes for the bear image and the condor image

## Use torch.topk method to get the most commmon categories on both bear and condor probability tensors

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

## INSERT TASK 3 CODE HERE

print('\nBear\n')
# Show top categories per image
top5_prob, top5_catid = None
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

print('\nCondor\n')
# Show top categories per image
top5_prob, top5_catid = None
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

## END TASK 3 CODE
