#!/usr/bin/env python
# coding: utf-8

import os,sys,inspect
import numpy as np
import argparse
import pandas as pd
import pdb
import torch
import torchvision, torchvision.transforms
import yaml
import random
from sklearn import metrics
from tqdm import tqdm as tqdm_base
sys.path.insert(0,"/ocean/projects/asc170022p/nmurali/projects/CounterfactualExplainer/MIMICCX-Chest-Explainer/Classifier/torchxrayvision_")
import torchxrayvision as xrv


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

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
        
def get_metrics(y_true, y_score, thres, class_names, prefix):
    y_true = np.nan_to_num(y_true, 0)
    y_pred = np.asarray(y_score > thres).astype(int)
    results = {}
    for i in range(0, y_true.shape[1]):
        task_r = {}
        task_r[prefix+'_acc'] = metrics.accuracy_score(y_true[:,i], y_pred[:,i]) 
        fpr, tpr, thresholds = metrics.roc_curve(y_true[:,i], y_score[:,i])
        task_r[prefix+'auc'] = metrics.auc(fpr, tpr)
        task_r[prefix+'cm'] = metrics.confusion_matrix(y_true[:,i], y_pred[:,i])
        task_r[prefix+'recall'] = metrics.recall_score(y_true[:,i], y_pred[:,i])
        task_r[prefix+'precision'] = metrics.precision_score(y_true[:,i], y_pred[:,i])
        results[class_names[i]] = task_r  
    return results

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', '-c', default='Configs/Classifier/NIH/NIH_test.yaml')
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
                                    'Classifier_Seed_'+str(config['seed'])+'_Dropout_'+str(config['drop_rate'])+
                                    '_LS_'+str(config['labelSmoothing']) + '_MU_'+str(config['mixUp'])+
                                    '_FL_'+str(config['focalLoss'])+'_'+str(config['data_file'].split('/')[-1].split('.')[0])
                                   )
config['name'] = config['dataset'] + '_' + str(config['size'])
config['class_names'] = config['class_names'].split(",")
# ============= Import ====================
sys.path.insert(0,os.path.join(main_dir,"Classifier"))
import train_utils
import datasets
import models
from temperature_scaling import ModelWithTemperature

# ============= Seed ====================    
np.random.seed(config['seed'])
random.seed(config['seed'])
torch.manual_seed(config['seed'])
if config['cuda']:
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ============= Dataset ====================
df = pd.read_csv(config['data_file'])
try:
    # df_train = df.loc[(df[config['column_name_split']]==1)]
    # train_inds = np.asarray(df_train.index)
    df_train = df.loc[(df[config['column_name_split']]==0)]
    test_inds = np.asarray(df_train.index)
    print("test: ", test_inds.shape)
except:
    print("The data_file don't have train column, during training we have randomly split the entire dataset to have 15% samples as validation set.")
    train_inds=np.load(os.path.join(config['output_dir'], 'train.npy'))
    test_inds=np.load(os.path.join(config['output_dir'], 'validation.npy'))

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

elif config['dataset'] == 'Stanford-CHEX':
    transforms = torchvision.transforms.Compose([
        #torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((config['size'], config['size'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(center_crop()),
        torchvision.transforms.Lambda(normalize())
    ])
    train_inds = datasets.CheX_Dataset(imgpath=config['imgpath'], csvpath=config['data_file'], class_names=config['class_names'], transform=transforms, seed=config['seed'])
    test_inds = datasets.CheX_Dataset(imgpath=config['imgpath'], csvpath=config['data_file_test'], class_names=config['class_names'], transform=transforms, seed=config['seed'])
    dataset = None

elif (config['dataset']=='MIMIC-CXR')or(config['dataset']=='NIH'):
    transforms = torchvision.transforms.Compose([
        #torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((config['size'], config['size'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(center_crop()),
        torchvision.transforms.Lambda(normalize())
    ])
    dataset = datasets.MIMIC_Dataset(csvpath=config['data_file'], class_names=config['class_names'], transform=transforms, seed=config['seed'])

if dataset is not None:
    # train_dataset = datasets.SubsetDataset(dataset, train_inds)
    
    test_dataset = datasets.SubsetDataset(dataset, test_inds)
else:
    # train_dataset = train_inds
    test_dataset = test_inds
# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=config['batch_size'],
#                                                shuffle=False,
#                                                num_workers=4, 
#                                                pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=config['batch_size'],
                                           shuffle=False,
                                           num_workers=4, 
                                           pin_memory=True)

num_cls= test_dataset.labels.shape[1]
print(num_cls, dataset.class_names)
# print(train_loader.dataset[0]["img"].shape, train_loader.dataset[0]["lab"].shape)
print("Test: ", len(test_loader.dataset))

# ============= Model ====================     
if config['ckpt_name'] == '':
    config['ckpt_name'] = config['dataset'] + "-" + config['model'] + "-" + config['name']  + '-best.pt'
print("Loading checkpoint: ", config['ckpt_name'])
weights_filename_local = os.path.join(config['output_dir'], config['ckpt_name'])

model = models.DenseNet(num_classes=config['num_classes'], in_channels=config['channel'], drop_rate = config['drop_rate'], weights = weights_filename_local, **models.get_densenet_params(config['model'])) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    model.cuda()
    
if config['TS']:
    if config['focalLoss']:
        new_model = ModelWithTemperature(model, config['alpha'])
    else:
        new_model = ModelWithTemperature(model, 0.0)
    new_model.set_temperature(test_loader)
    temperature = new_model.temperature.item()

save_dir = os.path.join(config['output_dir'], 'test')
if not os.path.exists(save_dir):
        os.makedirs(save_dir)
# ============= Testing ====================
counter = 0
sample_times = 50
for p in config["partition_name"]:
    if p == 'test':
        loader = test_loader
    else:
        # loader = train_loader
        raise('No train dataset defined in Test script')
        
    with torch.no_grad():
        t = tqdm(loader)
        for batch_idx, samples in enumerate(t):
            images = samples["img"].to(device)
            targets = samples["lab"].to(device)
            names = np.asarray(samples["file_name"])
                
            if config['drop_rate'] > 0:
                # init empty predictions
                y_ = np.zeros((sample_times, images.shape[0], num_cls)) 
                for sample_id in range(sample_times):
                    # save predictions from a sample pass
                    outputs = model(images)
                    outputs = np.asarray(outputs.detach().cpu())
                    if config['TS']:
                        outputs = outputs/temperature
                    outputs = 1/(1 + np.exp(-outputs)) 
                    
                    y_[sample_id] = outputs
                outputs = y_
            else:
                outputs = model(images)
                outputs = np.asarray(outputs.detach().cpu())
                if config['TS']:
                    outputs = outputs/temperature
                outputs = 1/(1 + np.exp(-outputs)) 
            
            targets = np.asarray(targets.detach().cpu())

            if batch_idx  == 0:
                all_targets = targets
                all_outputs = outputs
                all_names = names
            else:
                all_targets = np.append(all_targets, targets, axis=0)
                if config['drop_rate'] > 0:
                    all_outputs = np.append(all_outputs, outputs, axis=1)
                else:
                    all_outputs = np.append(all_outputs, outputs, axis=0)
                all_names = np.append(all_names, names, axis=0)

    print(config["partition_name"][counter], all_targets.shape, all_outputs.shape, all_names.shape)
    suffix = ''
    if config['TS']:
        suffix += 'TS'
    if config['drop_rate'] > 0:
        np.save(os.path.join(save_dir, 'y_true_' + suffix + config["partition_name"][counter] + '.npy'), all_targets)
        np.save(os.path.join(save_dir, 'y_pred_' + suffix + config["partition_name"][counter] + '.npy'), all_outputs)
        np.save(os.path.join(save_dir, 'names_' + suffix + config["partition_name"][counter] + '.npy'), all_names)
    else:  
        results = get_metrics(all_targets, all_outputs, 0.6, dataset.class_names, config["partition_name"][counter])
        df_results = pd.DataFrame(data = results)
        df_results.to_csv(os.path.join(save_dir, 'results_' + suffix + config["partition_name"][counter] + '.csv'))

        df_outcomes = pd.DataFrame()
        for i in range(0, all_targets.shape[1]):
            df_outcomes[dataset.class_names[i]] = all_targets[:,i]
            df_outcomes[dataset.class_names[i]+'_prob'] = all_outputs[:,i]
        df_outcomes['names'] = all_names
        df_outcomes.to_csv(os.path.join(save_dir, 'outcomes_' + suffix + config["partition_name"][counter] + '.csv'))
    
    counter += 1



