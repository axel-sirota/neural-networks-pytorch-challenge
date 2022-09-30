import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
model.eval()
image_batch = [Image.open('bear.png').convert('RGB'), Image.open('condor.png').convert('RGB')]
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = torch.stack([preprocess(image) for image in image_batch], 0).to(device)
print(input_tensor.size())

output = model(input_tensor)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0][:5])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities_bear = torch.nn.functional.softmax(output[0], dim=0)
probabilities_condor = torch.nn.functional.softmax(output[1], dim=0)

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

print('Bear')
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities_bear, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

print('Condor')
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities_condor, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
