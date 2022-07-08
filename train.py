import sys
import os
import logging
from model import MyCNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

import torchvision
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

from model import MyCNN
#from utils.utils import *

DEVICE = 'cuda'      # 'cuda' or 'cpu'
NUM_CLASSES = 7
DATASETS_NAMES = ['photo', 'art', 'cartoon', 'sketch']
CLASSES_NAMES = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']

# HYPERPARAMETER -------------------
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
GAMMA = 0.3          # Multiplicative factor for learning rate step-down
STEP_SIZE = 3

BATCH_SIZE = 128      
LR = 1e-3             # The initial Learning Rate
NUM_EPOCHS = 30       # Total number of training epochs (iterations over dataset)

MODE = '4C'           # '3A', '3B', '4A', '4C'
ALPHA = 0.25          # alpha
ALPHA_EXP = False


EVAL_ACCURACY_ON_TRAINING = False
SHOW_IMG = True       # if 'True' show images and graphs on output
SHOW_RESULTS = True   # if 'True' show images and graphs on output

# Define Data Preprocessing

# means and standard deviations ImageNet because the network is pretrained
means, stds = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# Define transforms to apply to each image
transf = transforms.Compose([ transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                              transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
])

# Define pacs data path
DATA_PATH = "/home/wangannan/practice/data/images"

PHOTO_PATH = os.path.join(DATA_PATH, 'photo')
ART_PATH = os.path.join(DATA_PATH, 'art_painting')
CARTOON_PATH = os.path.join(DATA_PATH,'cartoon')
SKETCH_PATH = os.path.join(DATA_PATH,'sketch')

# Prepare Pytorch train/test Datasets
photo_dataset = torchvision.datasets.ImageFolder(PHOTO_PATH, transform=transf)
art_dataset = torchvision.datasets.ImageFolder(ART_PATH, transform=transf)
cartoon_dataset = torchvision.datasets.ImageFolder(CARTOON_PATH, transform=transf)
sketch_dataset = torchvision.datasets.ImageFolder(SKETCH_PATH, transform=transf)

# Check dataset sizes
print(f"photo: {len(photo_dataset)}")
print(f"art: {len(art_dataset)}")
print(f"cartoon: {len(cartoon_dataset)}")
print(f"sketch: {len(sketch_dataset)}")

# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
photo_dataloader = DataLoader(photo_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
art_dataloader = DataLoader(art_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
cartoon_dataloader = DataLoader(cartoon_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
sketch_dataloader = DataLoader(sketch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)

# for img, _ in photo_dataloader : 
#   print(img.shape)  #(3, 227, 227)

def itr_merge(*itrs):
  for itr in itrs:
    for v in itr:
      yield v

train_dataloader = itr_merge(photo_dataloader, art_dataloader, cartoon_dataloader)
test_dataloader = sketch_dataloader

# Loading model 
net = MyCNN().to(DEVICE)
#print(net) #check size output layer OK

# Define loss function: CrossEntrpy for classification
criterion = nn.CrossEntropyLoss()

parameters_to_optimize = net.parameters()

optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

current_step = 0
accuracies_train = []
accuracies_validation = []
loss_class_list = []
loss_target_list = []
loss_source_list = []

for epoch in range(NUM_EPOCHS):
  
  net.train(True)

  print(f"--- Epoch {epoch+1}/{NUM_EPOCHS}, LR = {scheduler.get_last_lr()}")
  
  # Iterate over the dataset
  for source_images, source_labels in train_dataloader:
    source_images = source_images.to(DEVICE)
    source_labels = source_labels.to(DEVICE)    

    optimizer.zero_grad()
    
    outputs = net(source_images)          
    loss_class = criterion(outputs, source_labels)  
    loss_class_list.append(loss_class.item())
    loss_class.backward()

    optimizer.step()      
    
  with torch.no_grad():
    net.train(False)

    running_corrects_train = 0

    for images_train, labels_train in train_dataloader:
      # images, labels = next(iter(source_dataloader))
      images_train = images_train.to(DEVICE)
      labels_train = labels_train.to(DEVICE)

      # Forward Pass
      outputs_train = net(images_train)
      # Get predictions
      _, preds = torch.max(outputs_train.data, 1)

      # Update Corrects
      running_corrects_train += torch.sum(preds == labels_train.data).data.item()

    # Calculate Accuracy
    accuracy_train = running_corrects_train / float(len(train_dataloader)*(test_dataloader.batch_size))
    accuracies_train.append(accuracy_train)
    print('Accuracy on train (photo):', accuracy_train)

  # Step the scheduler
  current_step += 1
  scheduler.step() 

if SHOW_RESULTS: 
  print()
  print("Loss classifier")
  print(loss_class_list)

net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
net.train(False) # Set Network to evaluation mode

running_corrects = 0
for images, labels in tqdm(test_dataloader):
  images = images.to(DEVICE)
  labels = labels.to(DEVICE)

  # Forward Pass
  outputs = net(images)

  # Get predictions
  _, preds = torch.max(outputs.data, 1)

  # Update Corrects
  running_corrects += torch.sum(preds == labels.data).data.item()

# Calculate Accuracy
accuracy = running_corrects / float(len(art_dataset))

print('\nTest Accuracy (art painting): {} ({} / {})'.format(accuracy, running_corrects, len(art_dataset)))