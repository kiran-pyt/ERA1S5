
import torch
import torch.nn.functional as F
import models
#CODE BLOCK: 3
from torchvision import datasets, transforms     #Often images need to be transformed (turned into numbers/processed/augmented) before being used with a model, common image transformations are found here.
# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1407,), (0.4081,))
    ])
#CODE BLOCK: 4
train_data = datasets.MNIST('../inputs/data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../inputs/data', train=False, download=True, transform=train_transforms)
print(train_data)


