import matplotlib.pyplot as plt
import os, fnmatch
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
from collections import OrderedDict

user_input_gpu = input("Would you like to use a GPU to make a prediction? (Recommended) - Type 'Yes' or 'No': ")
user_input_model = input("Which model did you use to train the model? - Type 'vgg' or 'alexnet': ")

def find(pattern, path):
    """ Finds image within the directory."""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

image_dir = find('*.jpg', os.getcwd() + '/image_test')

if user_input_gpu.lower() == 'yes':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
if user_input_model == 'vgg':
    model = models.vgg16(pretrained=True)
    hidden_units = 25088
else: 
    model = models.alexnet(pretrained=True)
    hidden_units = 9216


train_dir = 'flowers/train'
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

def load_checkpoint(filepath):
    for param in model.parameters():
        param.requires_grad = False

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
    
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    width, height = im.size
    image = im.resize((256, 256))
    
    width, height = image.size
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
    
    image = (image-np.mean(image)) / np.std(image)
    
    return torch.Tensor(image.transpose())


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image[None,:,:,:].to(device)
    model.to(device)
    model.eval()

    ps = torch.exp(model(image))

    top_p, top_class = ps.topk(topk, dim=1)

    return top_p, top_class

model = load_checkpoint(user_input_model + '_checkpoint.pth')
top_p, top_class = predict(image_dir[0], model)

class_dict = train_data.class_to_idx
top_class = top_class.tolist()[0]
top_p = top_p.tolist()[0]
classes = [[j for j in class_dict if class_dict[j]==i][0] for i in top_class]

print("Top 5 Classes of images: {}".format(classes))
print("Top 5 probability of classes: {}".format([round(i, 2) for i in top_p]))

