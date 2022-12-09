from PIL import Image
from matplotlib import pyplot as plt
from os.path import join
from skimage.io import imread, imsave
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os,sys,os.path
import pandas as pd
import pydicom
import skimage
import collections
import pprint
import torch
import torch.nn.functional as F
import torchvision
import skimage.transform
import random

class Dataset():
    def __init__(self):
        pass
    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.class_names,counts))
    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()
    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")
        
class SubsetDataset(Dataset):
    def __init__(self, dataset, idxs=None):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.class_names = dataset.class_names
        
        self.idxs = idxs
        
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]
        
        self.csv = self.csv.reset_index(drop=True)
        
        if hasattr(self.dataset, 'which_dataset'):
            self.which_dataset = self.dataset.which_dataset[self.idxs]
    
    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n","\n  ")
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


    
class MIMIC_Dataset(Dataset):
    """
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S. 
    MIMIC-CXR: A large publicly available database of labeled chest radiographs. 
    arXiv preprint arXiv:1901.07042. 2019 Jan 21.
    
    https://arxiv.org/abs/1901.07042
    
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """
    def __init__(self, csvpath, class_names, transform, data_aug=None,
                 seed=0, unique_patients=True):

        super(MIMIC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.class_names = class_names
        
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.csv = self.csv.fillna(0)
        # self.views = ["PA"]
        # self.csv = self.csv[self.csv["ViewPosition"].isin(self.views)]

        # if unique_patients:
        #     self.csv = self.csv.groupby("patient_id").first().reset_index()
        # Get our classes.
        # healthy = self.csv["No Finding"] == 1
        self.labels = []
        for name in self.class_names:
            if name in self.csv.columns:
                # self.csv.loc[healthy, name] = 0
                mask = self.csv[name]   
            self.labels.append(mask.values)    
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        # make all the -1 values into 0 to keep things simple
        self.labels[self.labels == -1] = 0
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self), self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = str(self.csv.iloc[idx]["path"])
        img = Image.open(img_path)
        if len(img.getbands())>1:   # needed for NIH dataset as some images are RGBA 4 channel
            img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)            
        if self.data_aug is not None:
            img = self.data_aug(img)   

        return {"img":img, "lab":self.labels[idx], "idx":idx, "file_name" : img_path}

    
class CheX_Dataset(Dataset):
    """
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
    Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, 
    Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong, 
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, 
    Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng. https://arxiv.org/abs/1901.07031
    
    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/
    """
    def __init__(self, imgpath, csvpath, class_names, transform, data_aug=None,
                 seed=0, unique_patients=True):

        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        
        self.class_names = class_names
        
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = ["PA", "AP"]
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.csv = self.csv.fillna(0)              
        self.csv["view"] = self.csv["Frontal/Lateral"] # Assign view column 
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"] # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"}) # Rename Lateral with L  
        self.csv = self.csv[self.csv["view"].isin(self.views)] # Select the view 
         
        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat = '(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()
                   
        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.class_names:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
                
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        # make all the -1 values into 0 to keep things simple
        self.labels[self.labels == -1] = 0
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['Path'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        from PIL import Image
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.data_aug is not None:
            img = self.data_aug(img)
        return {"img":img, "lab":self.labels[idx], "idx":idx}    

    
class DIRTY_MNIST_Dataset(Dataset):
    def __init__(self, csvpath, class_names, transform, data_aug=None, seed=0):

        super(DIRTY_MNIST_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        
        self.class_names = class_names

        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        class_names_int = [int(i) for i in class_names]
        csv = pd.read_csv(self.csvpath)
        csv = csv.loc[csv['label'].isin(class_names_int)]
        self.csv = csv
        self.labels = []
        for name in self.class_names:
            if name in self.csv.columns:
                mask = self.csv[name]
                self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self),  self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = str(self.csv.iloc[idx]["names"])
        img = np.load(img_path)
        if self.transform is not None:
            img = self.transform(img)   
        if self.data_aug is not None:
            img = self.data_aug(img)
        return {"img":img, "lab":self.labels[idx], "idx":idx, "file_name" : img_path}
    
class HAM_Dataset(Dataset):
    def __init__(self, imgpath, csvpath, class_names, transform, data_aug=None,
                 seed=0, unique_patients=False):

        super(HAM_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.class_names = class_names    
        #["mel", "nv", "bcc", "akiec",   "bkl",  "df",  "vasc"]
        
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        
        self.labels = []
        for name in self.class_names:
            if name in self.csv.columns:
                mask = self.csv[name]
                self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        
        if unique_patients:
            self.csv = self.csv.groupby("lesion_id").first().reset_index()
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self),  self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            lesion_id = str(self.csv.iloc[idx]["lesion_id"])
            image_id = str(self.csv.iloc[idx]["image_id"])
            img_path = os.path.join(self.imgpath, image_id + '.jpg')
        except:
            img_path = str(self.csv.iloc[idx]["names"])
        
        try:
            img = imread(img_path)
        except:
            return {"img":None, "lab":None, "idx":idx}

        if self.transform is not None:
            img = self.transform(img)    
        if self.data_aug is not None:
            img = self.data_aug(img)
        return {"img":img, "lab":self.labels[idx], "idx":idx, "file_name" : img_path}
    
    
class HAM_NPY_Dataset(Dataset):
    def __init__(self, imgpath, csvpath, transform=None, data_aug=None,
                 flat_dir=True, seed=0, unique_patients=False):

        super(HAM_NPY_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255
        
        self.class_names = ["mel",
                            "nv",
                            "bcc",
                            "akiec",
                            "bkl",
                            "df",
                            "vasc"]
        
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        
        self.labels = []
        for pathology in self.class_names:
            if pathology in self.csv.columns:
                mask = self.csv[pathology]
                
                self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self),  self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img_path = str(self.csv.iloc[idx]["path"])
        try:
            img = np.load(img_path)
        except:
            return {"img":None, "lab":None, "idx":idx,"file_name" :img_path}
        if self.transform is not None:
            print('1',img.shape)
            img = torch.FloatTensor(img)
            print('2',img.shape, type(img))
            img = torchvision.transforms.ToPILImage()(img)
            print('3',img.size, type(img))
            img = self.transform(img)
            print('4',img.size, type(img), self.transform)
            img = torchvision.transforms.ToTensor()(img)
            print('5',img.shape, type(img))
        return {"img":img, "lab":self.labels[idx], "idx":idx, "file_name" : img_path}

    
class AFHQ_Dataset(Dataset):
    def __init__(self, csvpath, class_names, transform, data_aug=None, seed=0):

        super(AFHQ_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 1
        
        self.class_names = class_names
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        
        self.labels = []
        for name in self.class_names:
            if name in self.csv.columns:
                mask = self.csv[name]
                self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self),  self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):        
        img_path = str(self.csv.iloc[idx]["image-path"])
        try:
            img = imread(img_path)
        except:
            return {"img":None, "lab":None, "idx":idx}

        if self.transform is not None:
            img = self.transform(img)   
        if self.data_aug is not None:
            img = self.data_aug(img)
        return {"img":img, "lab":self.labels[idx], "idx":idx, "file_name" : img_path}
    
class CelebA(Dataset):
    def __init__(self, imgpath, csvpath, class_names, transform, data_aug=None, seed=0):

        super(CelebA, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.        
        self.imgpath = imgpath        
        self.transform = transform                
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.class_names = class_names
        self.labels = []
        for name in self.class_names:
            if name in self.csv.columns:
                mask = self.csv[name]
                self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        self.labels = (self.labels==1).astype(np.float32)   # convert any label which is not 1 to 0
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self),  self.data_aug)
    
    def __len__(self):
        # return number of images based on split (train, val or test)
        return len(self.csv)

    def __getitem__(self, idx):          
        
        img_path = os.path.join(self.imgpath, self.csv.iloc[idx]['image'])
        try:
            img = imread(img_path)
        except:
            return {"img":None, "lab":None, "idx":idx}

        img = self.transform(img)
                
        if self.data_aug is not None:
            img = self.data_aug(img)
        return {"img":img, "lab":self.labels[idx], "idx":idx, "file_name":img_path}
    
    
class ToPILImage(object):
    def __init__(self,channel = 1):
        self.channel = channel
        self.to_pil = transforms.ToPILImage(mode="F")

    def __call__(self, x):
        if self.channel == 1:
            return(self.to_pil(x[0]))
        else:
            return(self.to_pil(x))


# note this dataset converts -1 and 1 to 0 and 1
# transform=None => no transform is applied
# os.path.join(root_dir, df[img_field_path][idx]) forms the path to images
class AnyDataset(Dataset):
    def __init__(self, df=None, csv_path=None, root_dir='', img_path_field='path', return_labels=False, 
            transform=None, pathologies=None):
        super(AnyDataset, self).__init__()
               
        self.transform = transform
        self.root_dir = root_dir
        self.img_path_field = img_path_field
        
        if df is not None:
            self.df = df
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise Exception('Provide either df or csv_path')
        self.return_labels = return_labels
        
        if transform is not None:
            self.transform = transform
        else:
            # default transformation
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((256, 256)),                
                torchvision.transforms.ToTensor(),
            ])
            
        self.pathologies = pathologies
        
        if return_labels:
            self.labels = []
            for pathology in self.pathologies:
                if pathology in self.df.columns:
                    mask = self.df[pathology]
                    self.labels.append(mask.values)
            self.labels = np.asarray(self.labels).T
            self.labels = self.labels.astype(np.float32)
            self.labels = (self.labels==1).astype(np.float32)   # convert labels with -1 and 1, to 0 and 1
    
    def __len__(self):        
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df[self.img_path_field][idx])
        img = imread(img_path)

        if self.return_labels:                 
            img = self.transform(img)
            return {"x":img, "y":self.labels[idx], "idx":idx, "path":img_path}
        else:
            img = self.transform(img)
            return {"x":img,  "idx":idx, "path":img_path}