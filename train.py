'''
Coded by Ncuti Xavier
'''
from utils_fun import set_argument_parser, display_time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import OrderedDict

import os, random
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import argparse
import time


class Train_model():
    def __init__(self, model, data_dir='flowers', device="cuda", checkpoint='classifier.pth', 
    arch='vgg', epochs = 5, learning_rate=0.001, save_dir=None):
        
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'
        self.save_dir = save_dir
        self.arch = arch
        self.model = model
        
        self.learning_rate = learning_rate
        self.epochs = epochs 
        self.checkpoint = checkpoint
        
        self.data_loaders_fun()
        self.neural_network_fun()
        
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(),lr=self.learning_rate)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        
    def data_loaders_fun(self):
        data_transforms = {
    
            'training' : transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]),

            'validation' : transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]),

            'testing' : transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        }

        # TODO: Load the datasets with ImageFolder
        self.image_datasets = {
            'training' : datasets.ImageFolder(self.train_dir, transform=data_transforms['training']),
            'validation' : datasets.ImageFolder(self.valid_dir, transform=data_transforms['validation']),
            'testing' : datasets.ImageFolder(self.test_dir, transform=data_transforms['testing'])
        }

        # TODO: Using the image datasets and the trainforms, define the dataloaders
        self.dataloaders = {
            'training' : torch.utils.data.DataLoader(self.image_datasets['training'], batch_size=64, shuffle=True),
            'validation' : torch.utils.data.DataLoader(self.image_datasets['validation'], batch_size=64),
            'testing' : torch.utils.data.DataLoader(self.image_datasets['testing'], batch_size=64)
        }
        print("Passed")

    def neural_network_fun(self):
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        if self.arch == 'vgg':
            self.model = models.vgg19(pretrained=True)        
        elif self.arch == 'resnet':
            self.model = models.resnet18(pretrained=True)
        elif self.arch == 'alexnet':
            self.model = models.alexnet(pretrained = True)
            
            
        for param in self.model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('drop', nn.Dropout(p=0.4)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(4096, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))


        self.model.classifier = classifier
        
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        model = self.model.classifier.to(device)
        print(model)
    
    def validation(self, model, criterion, validation_loader):
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
        model.eval()
        accuracy = 0
        test_loss = 0

        for inputs, labels in iter(validation_loader):

            inputs = inputs.to(device)
            labels = labels.to(device) 

            output = model.forward(inputs)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output).data 
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        return test_loss/len(validation_loader), accuracy/len(validation_loader) 
    
    def train(self):
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        start_time = time.time()
        
        print_every = 40
        running_loss= 0
        steps = 0


        for epoch in range(self.epochs):
            
            print("_"*100)
            print("EPOCH {}/{}".format(epoch+1, self.epochs))
            print("."*100)
            
            self.model.train()
            self.model.to(device)
            
            for inputs, labels in iter(self.dataloaders['training']):
                
                inputs = inputs.to(device)
                labels = labels.to(device)  
                steps += 1 
                self.optimizer.zero_grad() 

                output = self.model.forward(inputs)
                loss = self.criterion(output, labels)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    validation_loss, accuracy = self.validation(self.model, self.criterion, self.dataloaders['validation'])

                    print("Training Loss: {:.2f} ".format(running_loss/print_every),
                          "Validation Loss: {:.2f} ".format(validation_loss),
                          "Validation Accuracy: {:.2f}".format(accuracy))

                    self.model.train()
                    running_loss = 0

            display_time("Epoch "+str(epoch+1),start_time)
            print("."*100)
        display_time("Training ",start_time)

        #save checkpoint
        self.model.class_to_idx = self.image_datasets['training'].class_to_idx
        checkpoint = {
            'arch': 'vgg19',
            'input_size': 25088,
            'output_size': 102,
            'batch_size': 64,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'classifier': self.model.classifier,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict(), 
            'class_to_idx': self.model.class_to_idx
        }
        torch.save(checkpoint, self.checkpoint)
        print("Model Saved")
 
    def test(self):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        start_time = time.time()
        correct = 0
        total = 0
        with torch.no_grad():
            # Puts model into validation mode
            self.model.eval()
            for inputs, labels in iter(self.dataloaders['testing']):  

                inputs = inputs.to(device) 
                labels = labels.to(device) 

                outputs = self.model.forward(inputs)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy_per = 100 * correct / total
        print("."*100)
        print("Accuracy on test set: {:.2f}%".format(accuracy_per)) 
        display_time("Testing ", start_time)
        print("."*100)
        
    
    
if __name__=="__main__":

    parser=set_argument_parser()
    parser.add_argument("data_dir", help="directory of flowers")
    args= parser.parse_args()

    if args.gpu:
        device='cuda'
    else:
        device='cpu'

    print("Data Directory: ",args.data_dir,"\nDevice used: ", device)
    
    
    train_model = Train_model(
        model = models.vgg19(pretrained=True),
        data_dir = args.data_dir,
        checkpoint = args.checkpoint if args.checkpoint else 'classifier.pth',
        arch = args.arch if args.arch else 'vgg19',
        epochs = args.epochs if args.epochs else 5,    
        learning_rate = args.learning_rate if args.learning_rate else 0.001,
        save_dir = args.save_dir if args.save_dir else ''
    )
    
    train_model.train()    
    train_model.test()