from importlib.abc import Loader
import torchvision
import torch
import sys, os
sys.path.insert(0,"/ocean/projects/asc170022p/nmurali/projects/CounterfactualExplainer/MIMICCX-Chest-Explainer/Classifier/torchxrayvision_")
from torchxrayvision.datasets import CelebA
sys.path.insert(0,'/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Classifier')
import models
from skimage.io import imread
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class AnyDataset(Dataset):
    def __init__(self, df=None, csv_path=None, return_labels=False, transform=None, seed=0, pathologies=None):
        super(AnyDataset, self).__init__()
        
        np.random.seed(seed)  # Reset the seed so all runs are the same.        
        self.transform = transform
        
        if df is not None:
            self.df = df
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise Exception('Provide either df or csv_path')
        self.return_labels = return_labels
        
        if transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor()            
            ])
        else:
            self.transform = transform 
            
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
        img_path = self.df['path'][idx]
        img = imread(img_path)

        if self.return_labels:                 
            img = self.transform(img)
            return {"img":img, "lab":self.labels[idx], "idx":idx, "file_name":img_path}
        else:
            img = self.transform(img)
            return {"img":img,  "idx":idx, "file_name":img_path}

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
        img = imread(img_path)

        if self.transform is not None:
            img = self.transform(img)   
        if self.data_aug is not None:
            img = self.data_aug(img)
        return {"img":img, "lab":self.labels[idx], "idx":idx, "file_name" : img_path}

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

class SKIN(Dataset):
    def __init__(self, file_names, transform=None, seed=0):
        super(SKIN, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.               
        if transform == None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor()            
            ])
        else:
            self.transform =transform
        self.file_names = file_names
    
    def __len__(self):
        return self.file_names.shape[0]

    def __getitem__(self, idx):          
        img_path = self.file_names[idx]
        img = imread(img_path)
        img = self.transform(img)
        return {"img":img,  "idx":idx, "file_name":img_path}

def get_dataloader(dataset, seed=1234):

    if dataset=='celeba':
        transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.Resize((256, 256)),
                        torchvision.transforms.CenterCrop(170),
                        torchvision.transforms.Resize((256, 256)),
                        torchvision.transforms.ToTensor()            
                    ])
        train_dataset = CelebA(mode='train', transform=transforms, pathologies=['Young','Smiling'])
        train_loader_in = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=8,
                                                shuffle=True,
                                                num_workers=4, 
                                                pin_memory=True,
                                                drop_last=True)

        val_dataset = CelebA(mode='val', transform=transforms, pathologies=['Young','Smiling'])
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=8,
                                                shuffle=False,
                                                num_workers=4, 
                                                pin_memory=True)

        transforms2 = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor()
        ])
        
        afhq_csv = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Data/afhq.csv'
        afhq_dataset = AFHQ_Dataset(csvpath=afhq_csv, class_names=['cat','dog'], transform=transforms2, seed=seed)
        train_loader_out = torch.utils.data.DataLoader(afhq_dataset,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    drop_last=True)
    elif dataset=='afhq':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor()
        ])

        csv_train = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/afhq_train.csv'
        csv_val = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/afhq_val.csv'

        train_dataset = AFHQ_Dataset(csvpath=csv_train, class_names=['cat','dog'], transform=transforms, seed=seed)
        val_dataset = AFHQ_Dataset(csvpath=csv_val, class_names=['cat','dog'], transform=transforms, seed=seed)

        train_loader_in = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=8,
                                                    shuffle=False,
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    drop_last=True)

        transforms2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.ToTensor()
        ])
        data_path_svhn = '/ocean/projects/asc170022p/singla/Datasets/MNIST_Data/SVHN'
        svhn_dataset = torchvision.datasets.SVHN(data_path_svhn, transform=transforms2, download=True)
        train_loader_out = torch.utils.data.DataLoader(
            svhn_dataset,
            batch_size=8,
            shuffle=True,
            drop_last=True,
        )
    elif dataset=='mnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),\
            torchvision.transforms.Resize((64,64)), 
            torchvision.transforms.ToTensor()
        ])

        train_csv = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Data/dirty_mnist_train.csv'
        val_csv = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Data/dirty_mnist_val.csv'
        train_dataset = DIRTY_MNIST_Dataset(csvpath=train_csv, transform=transforms, class_names=['0','1','2','3','4','5','6'], seed=seed)
        val_dataset = DIRTY_MNIST_Dataset(csvpath=val_csv, transform=transforms, class_names=['0','1','2','3','4','5','6'], seed=seed)

        train_loader_in = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=64,
                                                    shuffle=True,
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=64,
                                                    shuffle=False,
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    drop_last=True)

        # cifar10 ood
        transforms2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64,64)), 
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor()
        ])

        cifar_path = '/ocean/projects/asc170022p/singla/Datasets/MNIST_Data/cifar10'
        cifar_dataset = torchvision.datasets.CIFAR10(cifar_path, transform=transforms2,\
                                    train=True, download=True)
        train_loader_out = torch.utils.data.DataLoader(
                                                        cifar_dataset,
                                                        batch_size=64,
                                                        shuffle=True,
                                                        drop_last=True,
                                                    )
    elif dataset=='skin':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),\
            torchvision.transforms.Resize((256,256)), 
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.ToTensor()
        ])

        imgpath = '/ocean/projects/asc170022p/singla/Datasets/HAM10000/imgs'
        train_csv = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/skin_train.csv'
        val_csv = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/skin_val.csv'
        train_dataset = HAM_Dataset(imgpath=imgpath,csvpath=train_csv,class_names=['nv','mel_or_bkl'],unique_patients=False, transform=transforms, seed=seed)
        val_dataset = HAM_Dataset(imgpath=imgpath,csvpath=val_csv,class_names=['nv','mel_or_bkl'],unique_patients=False, transform=transforms, seed=seed)

        train_loader_in = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=8,
                                                    shuffle=False,
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    drop_last=True)

        transforms2 = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor()
        ])

        csv_ood = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/afhq_train.csv'

        ood_dataset = AFHQ_Dataset(csvpath=csv_ood, class_names=['cat','dog'], transform=transforms2, seed=seed)

        train_loader_out = torch.utils.data.DataLoader(ood_dataset,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    drop_last=True)

    else:
        raise('Invalid dataset name!!')
    
    return train_loader_in, train_loader_out, val_loader


def get_model(dataset):

    if dataset=='celeba':
        net = models.DenseNet(num_classes=2, in_channels=3, drop_rate=0.0,**models.get_densenet_params('densenet121')) 
    elif dataset=='afhq' or dataset=='skin':
        net = models.DenseNet(num_classes=2, in_channels=3, drop_rate=0.0,**models.get_densenet_params('densenet169')) 
    elif dataset=='mnist':
        net = models.DenseNet(num_classes=7, in_channels=1, drop_rate=0.0,**models.get_densenet_params('densenet121')) 
    else:
        raise('Invalid dataset name!!')

    return net

def get_ood_loaders(name, type):

    if name=='celeba':
        if type=='aid':
            aid_csv_path = '/ocean/projects/asc170022p/nmurali/projects/CounterfactualExplainer/MIMICCX-Chest-Explainer/experiments/sumedha_code/data/celeba_aid.csv'
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(170),
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor()            
            ])
            aid_dataset = AnyDataset(csv_path=aid_csv_path, transform=transforms)
            loader = torch.utils.data.DataLoader(aid_dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           num_workers=4, 
                                           pin_memory=True,
                                           drop_last=True)
        elif type=='nood':
            utk_nood_path = '/ocean/projects/asc170022p/nmurali/data/utk_face/utk_ood.csv'
            transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.Resize((256, 256)),
                            torchvision.transforms.ToTensor()            
                        ])
            utk_nood_dataset = AnyDataset(csv_path=utk_nood_path, transform=transforms)
            loader = torch.utils.data.DataLoader(utk_nood_dataset,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    drop_last=True)
        elif type=='food1':
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((256, 256)), 
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor()
            ])

            csv_train = '/ocean/projects/asc170022p/singla/Datasets/AFHQ/data/wild.csv'

            train_dataset = AFHQ_Dataset(csvpath=csv_train, class_names=['wild'], transform=transforms, seed=1234)

            loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=8,
                                                        shuffle=True,
                                                        num_workers=4, 
                                                        pin_memory=True,
                                                        drop_last=True)

        elif type=='food2':
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256,256)), 
                torchvision.transforms.ToTensor()
            ])

            cifar_path = '/ocean/projects/asc170022p/singla/Datasets/MNIST_Data/cifar10'
            cifar_dataset = torchvision.datasets.CIFAR10(cifar_path, transform=transforms,\
                                        train=True, download=True)
            loader = torch.utils.data.DataLoader(
                                                            cifar_dataset,
                                                            batch_size=8,
                                                            shuffle=True,
                                                            drop_last=True,
                                                        )
    elif name=='afhq':
        if type=='aid':
            aid_csv_path = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/afhq_aid.csv'
            transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor()
        ])
            aid_dataset = AnyDataset(csv_path=aid_csv_path, transform=transforms)
            loader = torch.utils.data.DataLoader(aid_dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           num_workers=4, 
                                           pin_memory=True,
                                           drop_last=True)
        elif type=='nood':
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((256, 256)), 
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor()
            ])

            csv_train = '/ocean/projects/asc170022p/singla/Datasets/AFHQ/data/wild.csv'

            train_dataset = AFHQ_Dataset(csvpath=csv_train, class_names=['wild'], transform=transforms, seed=1234)

            loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=8,
                                                        shuffle=True,
                                                        num_workers=4, 
                                                        pin_memory=True,
                                                        drop_last=True)
        elif type=='food1':
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256,256)), 
                torchvision.transforms.ToTensor()
            ])

            cifar_path = '/ocean/projects/asc170022p/singla/Datasets/MNIST_Data/cifar10'
            cifar_dataset = torchvision.datasets.CIFAR10(cifar_path, transform=transforms,\
                                        train=True, download=True)
            loader = torch.utils.data.DataLoader(
                                                            cifar_dataset,
                                                            batch_size=8,
                                                            shuffle=True,
                                                            drop_last=True,
                                                        )
        elif type=='food2':
            transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.Resize((256, 256)),
                            torchvision.transforms.CenterCrop(170),
                            torchvision.transforms.Resize((256, 256)),
                            torchvision.transforms.ToTensor()            
                        ])

            val_dataset = CelebA(mode='val', transform=transforms, pathologies=['Young','Smiling'])
            loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=8,
                                                    shuffle=False,
                                                    num_workers=4, 
                                                    pin_memory=True)

    elif name=='skin':
        if type=='aid':
            aid_csv_path = '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/skin_aid.csv'
            transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.ToTensor()
        ])
            aid_dataset = AnyDataset(csv_path=aid_csv_path, transform=transforms)
            loader = torch.utils.data.DataLoader(aid_dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           num_workers=4, 
                                           pin_memory=True,
                                           drop_last=True)
        elif type=='nood':
            df_file_name = '/ocean/projects/asc170022p/singla/Datasets/HAM10000/HAM10000_metadata_nv.csv'
            df_file_name_all = '/ocean/projects/asc170022p/singla/Datasets/HAM10000/HAM10000_metadata'
            imgpath = '/ocean/projects/asc170022p/singla/Datasets/HAM10000/imgs/'
            df_nv = pd.read_csv(df_file_name)
            df_all = pd.read_csv(df_file_name_all)
            No_nv_image_id = set(list(df_all['image_id'])) - set(list(df_nv['image_id']))
            df_no_nv = df_all.loc[df_all['image_id'].isin(list(No_nv_image_id))]
            df_no_nv['names'] = imgpath + df_no_nv['image_id']+'.jpg'
            ood_names = np.asarray(df_no_nv['names'])
            ood_dataset = SKIN(file_names=ood_names)
            loader = torch.utils.data.DataLoader(
                ood_dataset,
                batch_size=8,
                drop_last=True,
                shuffle=True
            )
        elif type=='food2':
            df_file_name = '/ocean/projects/asc170022p/singla/Datasets/HAM10000/HAM10000_metadata_nv.csv'
            df_file_name_all = '/ocean/projects/asc170022p/singla/Datasets/HAM10000/HAM10000_metadata'
            imgpath = '/ocean/projects/asc170022p/singla/Datasets/HAM10000/imgs/'
            df_nv = pd.read_csv(df_file_name)
            df_all = pd.read_csv(df_file_name_all)
            No_nv_image_id = set(list(df_all['image_id'])) - set(list(df_nv['image_id']))
            df_no_nv = df_all.loc[df_all['image_id'].isin(list(No_nv_image_id))]
            df_no_nv['names'] = imgpath + df_no_nv['image_id']+'.jpg'
            ood_names = np.asarray(df_no_nv['names'])
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomCrop(size= 60),
                torchvision.transforms.Resize((256,256)),
                torchvision.transforms.ToTensor()
            ])
            ood_dataset = SKIN(file_names=ood_names, transform = transforms)
            loader = torch.utils.data.DataLoader(
                ood_dataset,
                batch_size=8,
                drop_last=True,
                shuffle=True
            )
        elif type=='food1':
            transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.Resize((256, 256)),
                            torchvision.transforms.CenterCrop(170),
                            torchvision.transforms.Resize((256, 256)),
                            torchvision.transforms.ToTensor()            
                        ])

            val_dataset = CelebA(mode='val', transform=transforms, pathologies=['Young','Smiling'])
            loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=8,
                                                    shuffle=False,
                                                    num_workers=4, 
                                                    pin_memory=True)

    elif name=='mnist':
        if type=='aid':
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),\
                torchvision.transforms.Resize((64, 64)), 
                torchvision.transforms.ToTensor()
            ])
            df_test_AM = '/ocean/projects/asc170022p/singla/Datasets/MNIST_Data/ambiguous_mnist_Test_0_6_1000_each.csv'
            test_dataset_AM = DIRTY_MNIST_Dataset(csvpath=df_test_AM, transform=transforms, class_names=['0','1','2','3','4','5','6'])
            loader = torch.utils.data.DataLoader(
                test_dataset_AM,
                batch_size=16,
                shuffle=False,
                drop_last=True,
            )
        elif type=='nood':
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),\
                torchvision.transforms.Resize((64, 64)), 
                torchvision.transforms.ToTensor()
            ])
            df_train_ood = '/ocean/projects/asc170022p/singla/Datasets/MNIST_Data/mnist_Train.csv'
            train_ood_dataset = DIRTY_MNIST_Dataset(csvpath=df_train_ood, transform=transforms, \
                                                                class_names=['7','8','9'])
            loader = torch.utils.data.DataLoader(
                train_ood_dataset,
                batch_size=16,
                shuffle=True,
                drop_last=True,
            )
        elif type=='food1':
            transforms2 = torchvision.transforms.Compose([
                torchvision.transforms.Resize((64, 64)), 
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor()
            ])
            data_path_svhn = '/ocean/projects/asc170022p/singla/Datasets/MNIST_Data/SVHN'
            svhn_dataset = torchvision.datasets.SVHN(data_path_svhn, transform=transforms2, download=True)
            loader = torch.utils.data.DataLoader(
                svhn_dataset,
                batch_size=16,
                shuffle=True,
                drop_last=True,
            )
        elif type=='food2':
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((64, 64)), 
                torchvision.transforms.ToTensor()
            ])

            data_path_Fminst = '/ocean/projects/asc170022p/singla/Datasets/MNIST_Data/FahionMNIST'
            Fminst_train_dataset = torchvision.datasets.FashionMNIST(data_path_Fminst, transform=transforms,\
                                        train=True, download=True)
            loader = torch.utils.data.DataLoader(
                Fminst_train_dataset,
                batch_size=16,
                shuffle=False,
                drop_last=True,
            )

    return loader
