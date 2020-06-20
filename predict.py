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
import json
import argparse
import time

class Predict_model():
    def __init__(self, model, arch="vgg19", checkpoint="classifier.pth", 
    device="cuda", image_path=None, cat_to_name="cat_to_name.json"):
        
        self.arch = arch
        self.checkpoint = checkpoint
        self.device = device
        self.image_path = image_path
        
        with open(cat_to_name, 'r') as f:
            self.cat_to_name = json.load(f)
        
        self.model = model       
        self.loaded_model = self.load_checkpoint()

     
    def load_checkpoint(self):
        
        self.checkpoint = torch.load(self.checkpoint)
        
        if self.arch == 'vgg':
            self.model = models.vgg19(pretrained=True)        
        elif self.arch == 'resnet':
            self.model = models.resnet18(pretrained=True)
        elif self.arch == 'alexnet':
            self.model = models.alexnet(pretrained = True)
        
        self.model.classifier = self.checkpoint['classifier']
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.class_to_idx = self.checkpoint['class_to_idx']

        optimizer = self.checkpoint['optimizer']

        for param in self.model.parameters():
            param.requires_grad = False
            
        return self.model
    
    
    def process_image(self, img):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        img = Image.open(img)
        img.resize((256, 256))
        crop_size = 224

        val = 0.5*(256-crop_size)
        img = img.crop((val, val, 256-val, 256-val))
        img = np.array(img)/255

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        return img.transpose(2,0,1)
    
    def predict(self, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        self.model.to(device)
        self.model.eval()

        processed_image = self.process_image(self.image_path)
        image = torch.from_numpy(np.array([processed_image])).float()
        image = image.to(device)

        output = self.model.forward(image)
        probabilities = torch.exp(output).data
        probs = torch.topk(probabilities, topk)[0].tolist()[0]
        idx = torch.topk(probabilities, topk)[1].tolist()[0]

        index = []
        for i in range(len(self.model.class_to_idx.items())):
            index.append(list(self.model.class_to_idx.items())[i][0])

        labels = []
        for i in range(topk):
            labels.append(index[idx[i]])
        
        flowers = [self.cat_to_name[label] for label in labels] 
        
        return probs, flowers, labels
    
    def display_results(self, top_k):
        
        probabilities, flowers, labels = self.predict(top_k)  
        print("_"*70)
        print("FLOWER && PROBABILITY")
        print("_"*70)
        for i in range(len(flowers)):
            print(f"{flowers[i]} => {probabilities[i]:.2f}")
        print("_"*70)
        
if __name__=="__main__":

    parser=set_argument_parser()
    parser.add_argument("img_path", help="image path")
    args = parser.parse_args()

    if args.gpu:
        device='cuda'
    else:
        device='cpu'
    
    
    predict_model = Predict_model(
        model = models.vgg19(pretrained=True),
        arch = args.arch if args.arch else 'vgg19',        
        checkpoint = args.checkpoint if args.checkpoint else 'classifier.pth',
        device = device,
        image_path = args.img_path,
        cat_to_name = args.category_names if args.category_names else 'cat_to_name.json'
    )
    
    if args.top_k is not None:
        predict_model.display_results(top_k = args.top_k) 
    else:
        predict_model.display_classes(5)