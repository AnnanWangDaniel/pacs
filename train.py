import sys
import os
import logging
from model import MyCNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

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

BATCH_SIZE = 256      
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

# check dimensions of images
# cnt = 0
# for img, _ in dataloader : 
#   print(img.shape)
#   cnt+=1
# print(cnt)

### Prepare Network for training

cudnn.benchmark # Calling this optimizes runtime

if MODE == None :
  raise RuntimeError("Select a MODE")
elif MODE == '3A':  
  # 3A) SENZA DANN	
  USE_DOMAIN_ADAPTATION = False
  CROSS_DOMAIN_VALIDATION = False 
  USE_VALIDATION = False
  ALPHA = None
  transfer_set = None
elif MODE == '3B' : 
  # 3B) Train DANN on Photo and test on Art painting with DANN adaptation
  USE_DOMAIN_ADAPTATION = True 
  transfer_set = "art painting"
elif MODE == '4A':
  # 4A) Run a grid search on Photo to Cartoon and Photo to Sketch, without Domain Adaptation, and average results for each set of hyperparameters
  transfer_set = 'sketch' # Photo to 'cartoon' or 'sketch'
  USE_VALIDATION = True   # validation on transfer_set
  USE_DOMAIN_ADAPTATION = False
  CROSS_DOMAIN_VALIDATION = False 
  ALPHA = None
  # 4B) when testing
elif MODE == '4C':
  # 4C) Run a grid search on Photo to Cartoon and Photo to Sketch, with Domain Adaptation, and average results for each set of hyperparameters
  USE_VALIDATION = True   # validation on transfer_set
  USE_DOMAIN_ADAPTATION = True
  CROSS_DOMAIN_VALIDATION = True 
  # edit the following hyperparams:
  transfer_set = 'sketch' # Photo to 'cartoon' or 'sketch'


EVAL_ACCURACY_ON_TRAINING = False
SHOW_RESULTS = True

source_dataloader = photo_dataloader
test_dataloader = art_dataloader

# Loading model 
net = MyCNN().to(DEVICE)    
#print(net) #check size output layer OK

# Define loss function: CrossEntrpy for classification
criterion = nn.CrossEntropyLoss()

# Choose parameters to optimize
parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet

# Define optimizer: updates the weights based on loss (SDG with momentum)
optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Define scheduler -> step-down policy which multiplies learning rate by gamma every STEP_SIZE epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

if USE_DOMAIN_ADAPTATION and ALPHA == None :
  raise RuntimeError("To use domain adaptation you must define parameter ALPHA")

if transfer_set == 'cartoon':
  target_dataloader = cartoon_dataloader
elif transfer_set == 'sketch':
  target_dataloader = sketch_dataloader
else :
  target_dataloader = test_dataloader # art_dataloader

### TRAIN

current_step = 0
accuracies_train = []
accuracies_validation = []
loss_class_list = []
loss_target_list = []
loss_source_list = []

# Start iterating over the epochs
for epoch in range(NUM_EPOCHS):
  
  net.train(True)

  print(f"--- Epoch {epoch+1}/{NUM_EPOCHS}, LR = {scheduler.get_last_lr()}")
  
  # Iterate over the dataset
  for source_images, source_labels in source_dataloader:
    source_images = source_images.to(DEVICE)
    source_labels = source_labels.to(DEVICE)    

    optimizer.zero_grad() # Zero-ing the gradients
    
    # STEP 1: train the classifier
    outputs = net(source_images)          
    loss_class = criterion(outputs, source_labels)  
    loss_class_list.append(loss_class.item())

    # if current_step % LOG_FREQUENCY == 0:
    #   print('Step {}, Loss Classifier {}'.format(current_step+1, loss_class.item()))                
    loss_class.backward()  # backward pass: computes gradients

    # Domain Adaptation (Cross Domain Validation)
    if USE_DOMAIN_ADAPTATION :

      # Load target batch
      target_images, target_labels = next(iter(target_dataloader))
      target_images = target_images.to(DEVICE) 
      
      # if ALPHA_EXP : 
      #   # ALPHA exponential decaying as described in the paper
      #   p = float(i + epoch * len_dataloader) / NUM_EPOCHS / len_dataloader
      #   ALPHA = 2. / (1. + np.exp(-10 * p)) - 1
    
      # STEP 2: train the discriminator: forward SOURCE data to Gd          
      outputs = net.forward(source_images, alpha=ALPHA)
      # source's label is 0 for all data    
      labels_discr_source = torch.zeros(BATCH_SIZE, dtype=torch.int64).to(DEVICE)
      loss_discr_source = criterion(outputs, labels_discr_source)  
      loss_source_list.append(loss_discr_source.item())         
      # if current_step % LOG_FREQUENCY == 0:
      #   print('Step {}, Loss Discriminator Source {}'.format(current_step+1, loss_discr_source.item()))
      loss_discr_source.backward()

      # STEP 3: train the discriminator: forward TARGET to Gd          
      outputs = net.forward(target_images, alpha=ALPHA)           
      labels_discr_target = torch.ones(BATCH_SIZE, dtype=torch.int64).to(DEVICE) # target's label is 1
      loss_discr_target = criterion(outputs, labels_discr_target)    
      loss_target_list.append(loss_discr_target.item())     
      # if current_step % LOG_FREQUENCY == 0:
        # print('Step {}, Loss Discriminator Target {}'.format(current_step+1, loss_discr_target.item()))
      loss_discr_target.backward()    #update gradients 

    optimizer.step() # update weights based on accumulated gradients          
    
  # --- Accuracy on training
  if EVAL_ACCURACY_ON_TRAINING:
    with torch.no_grad():
      net.train(False)

      running_corrects_train = 0

      for images_train, labels_train in source_dataloader:
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
    accuracy_train = running_corrects_train / float(len(source_dataloader)*(target_dataloader.batch_size))
    accuracies_train.append(accuracy_train)
    print('Accuracy on train (photo):', accuracy_train)
    
  # --- VALIDATION SET
  if USE_VALIDATION : 
    # now train is finished, evaluate the model on the target dataset 
    net.train(False) # Set Network to evaluation mode
      
    running_corrects = 0
    for images, labels in target_dataloader:
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)
      
      outputs = net(images)
      _, preds = torch.max(outputs.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float( len(target_dataloader)*(target_dataloader.batch_size) )
    accuracies_validation.append(accuracy)
    print(f"Accuracy on validation ({transfer_set}): {accuracy}")

  # Step the scheduler
  current_step += 1
  scheduler.step() 

if SHOW_RESULTS: 
  print()
  print("Loss classifier")
  print(loss_class_list)
  if USE_DOMAIN_ADAPTATION : 
    print("\nLoss discriminator source")
    print(loss_source_list)
    print("\nLoss discriminator target")
    print(loss_target_list)

### TEST

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

### Print results
if USE_VALIDATION : 
  print(f"Validation on:  {transfer_set}")
  print(f"accuracy_valid: {accuracies_validation[-1]:.4f}")
print(f"Test accuracy:  {accuracy:.4f}")
print(f"Val on {transfer_set}, LR = {LR}, ALPHA = {ALPHA}, BATCH_SIZE = {BATCH_SIZE}")

if USE_DOMAIN_ADAPTATION :
  # Plot losses 
  plotLosses(loss_class_list, loss_source_list, loss_target_list, n_epochs=len(loss_class_list), show=SHOW_IMG)