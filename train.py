import matplotlib.pyplot as plt

import numpy as np
import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
from collections import OrderedDict

user_input_gpu = input("Would you like to use a GPU? (Recommended) - Type 'Yes' or 'No': ")
user_input_model = input("Which model would you like to use? - Type 'vgg' or 'alexnet': ")
learning_rate = float(input("Set the learning rate (0.0001 is recommended: "))
epochs = int(input("Set the number of epochs: "))

if user_input_model == 'alexnet':
    model = models.alexnet(pretrained=True)
    hidden_units = 9216
else: 
    # VGG set as default model
    model = models.vgg16(pretrained=True)
    hidden_units = 25088
    
    
if user_input_gpu.lower() == 'no':
    device = torch.device("cpu")
else:
    # GPU set as default value 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
   

for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc_1', nn.Linear(hidden_units, 512)),
                                 ('relu',nn.ReLU()),
                                 ('dropout_1', nn.Dropout(0.4)),
                                 ('fc_2', nn.Linear(512, 256)),
                                 ('relu',nn.ReLU()),
                                 ('dropout_2', nn.Dropout(0.4)),
                                 ('fc_3', nn.Linear(256, 128)),
                                 ('relu',nn.ReLU()),
                                 ('dropout_4', nn.Dropout(0.4)),
                                 ('fc_4', nn.Linear(128, 102)),
                                 ('output', nn.LogSoftmax(dim=1))]))
    
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device);

steps = 0
running_loss = 0
print_every = 60
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}    "
                  f"Train loss: {running_loss/print_every:.3f}    "
                  f"Validation loss: {test_loss/len(validationloader):.3f}    "
                  f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()
            
if user_input_model.lower() == "vgg":
    torch.save(model.state_dict(), 'vgg_checkpoint.pth')
else:
    torch.save(model.state_dict(), 'alexnet_checkpoint.pth')