#!/usr/bin/env python
# coding: utf-8

import os,sys,inspect
import numpy as np
import argparse
import pandas as pd
import torch
import torchvision, torchvision.transforms
import yaml
import random

class expand_greyscale(object):
    def __init__(self):
        self.num_target_channels = 3
    def __call__(self, tensor):
        channels = tensor.shape[0]        
        if channels == self.num_target_channels:
            return tensor
        elif channels == 1:
            color = tensor.expand(3, -1, -1)
            return color

class center_crop(object):
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y,x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]
    
    def __call__(self, img):
        return self.crop_center(img)

class normalize(object):
    def normalize_(self, img, maxval=255):
        img = (img)/(maxval)
        return img
    
    def __call__(self, img):
        return self.normalize_(img)
    
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', '-c', default='Configs/Classifier/NIH/NIH_pneum.yaml')
parser.add_argument(
'--main_dir', '-m', default='/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22')    
args = parser.parse_args()
main_dir = args.main_dir
# ============= Load config =============
config_path = os.path.join(main_dir, args.config)
config = yaml.safe_load(open(config_path)) 
print("Training Configuration: ")
print(config)  
config['output_dir'] = os.path.join(main_dir,
                                    config['output_dir'],
                                    config['dataset'],
                                    config['expt_name'])
config['name'] =config['dataset'] + '_' + str(config['size'])
config['class_names'] = config['class_names'].split(",")
# ============= Import ====================
sys.path.insert(0,os.path.join(main_dir,"Classifier"))
import train_utils
import datasets
import models
# ============= Dataset ====================
df = pd.read_csv(config['data_file'])
# df = df.sample(frac=1).head(250)
try:
    df_train = df.loc[(df[config['column_name_split']]==1)]
    train_inds = np.asarray(df_train.index)
    df_train = df.loc[(df[config['column_name_split']]==0)]
    test_inds = np.asarray(df_train.index)
    print("train: ", train_inds.shape, "test: ", test_inds.shape)
except:
    print("The data_file doesn't have a train column, hence we will randomly split the entire dataset to have 15% samples as validation set.")
    train_inds=np.empty([])
    test_inds=np.empty([])

if config['dataset'] == 'AFHQ':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),\
        torchvision.transforms.Resize((config['size'], config['size'])), 
        torchvision.transforms.RandomHorizontalFlip(p=config['data_aug_hf']),
        torchvision.transforms.ToTensor()
    ])
    
    dataset = datasets.AFHQ_Dataset(csvpath=config['data_file'], class_names=config['class_names'], transform=transforms, seed=config['seed'])
elif config['dataset'] == 'HAM':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),\
        torchvision.transforms.Resize((config['size'], config['size'])), 
        torchvision.transforms.RandomHorizontalFlip(p=config['data_aug_hf']),
        torchvision.transforms.RandomVerticalFlip(p=config['data_aug_hf']),
        torchvision.transforms.ToTensor()
    ])
    dataset = datasets.HAM_Dataset(imgpath=config['imgpath'],csvpath=config['data_file'],class_names=config['class_names'],unique_patients=False, transform=transforms, seed=config['seed'])

elif config['dataset'] == 'Dirty_MNIST':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),\
        torchvision.transforms.Resize((config['size'], config['size'])), 
        torchvision.transforms.ToTensor()
    ])
    train_inds = datasets.DIRTY_MNIST_Dataset(csvpath=config['data_file'], transform=transforms, class_names=config['class_names'], seed=config['seed'])
    test_inds = datasets.DIRTY_MNIST_Dataset(csvpath=config['data_file_test'], transform=transforms, class_names=config['class_names'], seed=config['seed'])
    dataset = None

elif config['dataset'] == 'CelebA':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((config['size'], config['size'])),
        torchvision.transforms.CenterCrop(config['center_crop']),
        torchvision.transforms.Resize((config['size'], config['size'])),
        torchvision.transforms.RandomHorizontalFlip(p=config['data_aug_hf']),
        torchvision.transforms.ToTensor()
    ])
    dataset = datasets.CelebA(imgpath=config['imgpath'],  csvpath=config['data_file'], class_names=config['class_names'], transform=transforms, seed=config['seed'])

# elif config['dataset'] == 'Stanford-CHEX':
#     transforms = torchvision.transforms.Compose([
#         #torchvision.transforms.ToPILImage(),
#         torchvision.transforms.Resize((config['size'], config['size'])),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Lambda(center_crop()),
#         torchvision.transforms.Lambda(normalize())
#     ])
#     train_inds = datasets.CheX_Dataset(imgpath=config['imgpath'], csvpath=config['data_file'], class_names=config['class_names'], transform=transforms, seed=config['seed'])
#     test_inds = datasets.CheX_Dataset(imgpath=config['imgpath'], csvpath=config['data_file_test'], class_names=config['class_names'], transform=transforms, seed=config['seed'])
#     dataset = None

elif (config['dataset']=='MIMIC-CXR') or (config['dataset']=='NIH') or (config['dataset']=='Chex_MIMIC') or (config['dataset']=='Chexpert'):
    transforms = torchvision.transforms.Compose([
        #torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((config['size'], config['size'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(center_crop()),
        torchvision.transforms.Lambda(normalize())
    ])
    dataset = datasets.MIMIC_Dataset(csvpath=config['data_file'], class_names=config['class_names'], transform=transforms, seed=config['seed'])
    
# ============= Seed ====================    
np.random.seed(config['seed'])
random.seed(config['seed'])
torch.manual_seed(config['seed'])
if config['cuda']:
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ============= Model ==================== 
model = models.DenseNet(num_classes=config['num_classes'], in_channels=config['channel'], drop_rate = config['drop_rate'],**models.get_densenet_params(config['model'])) 
# ============= Training ====================
train_utils.train(model, dataset, config, train_inds, test_inds)

print("Done")