from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import os
import numpy as np
import sys
import skimage
import skimage.transform
import warnings
import torchvision, torchvision.transforms
from skimage.io import imread, imsave



class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

    
# source: https://github.com/yunjey/stargan/blob/master/data_loader.py#L45
class CelebADataset(Dataset):
    def __init__(self, path, resolution=256, CenterCrop=170):
        super().__init__()
        self.folder = path #csv_file_name
        import pandas as pd
        df = pd.read_csv(path)
        try:
            df['temp'] = df['image-path']
        except:
            df['temp'] = df['names']
        self.paths = list(np.asarray(df['temp']))
        print("Number of files: ", len(self.paths))
        self.length = len(self.paths)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),\
            transforms.Resize((resolution, resolution)),
            transforms.CenterCrop(CenterCrop),
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor()
        ])


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = imread(path)
        img = self.transform(img)
        return img    
    
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
    
        
class XRayDataset(Dataset):
    def __init__(self, path, resolution=256):
        super().__init__()
        self.folder = path #csv_file_name
        import pandas as pd
        df = pd.read_csv(path)
        try:
            self.paths = list(np.asarray(df['lateral_512_jpeg']))
        except:
            self.paths = list(np.asarray(df['names']))
        print("Number of files: ", len(self.paths))
        self.length = len(self.paths)

        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Lambda(center_crop()),
            transforms.Lambda(expand_greyscale())#,
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = self.transform(img)
        return img

    
class SkinDataset(Dataset):
    def __init__(self, path,imgpath, resolution=256):
        super().__init__()
        self.folder = path #csv_file_name
        import pandas as pd
        df = pd.read_csv(path)
        try:
            df['temp'] = imgpath + '/' + df['image_id'] + '.jpg'
        except:
            df['temp'] = df['names']
        self.paths = list(np.asarray(df['temp']))
        print("Number of files: ", len(self.paths))
        self.length = len(self.paths)

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor()
        ])


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = self.transform(img)
        return img

class AFHQDataset(Dataset):
    def __init__(self, path, resolution=256):
        super().__init__()
        self.folder = path #csv_file_name
        import pandas as pd
        df = pd.read_csv(path)
        try:
            df['temp'] = df['image-path']
        except:
            df['temp'] = df['names']
        self.paths = list(np.asarray(df['temp']))
        print("Number of files: ", len(self.paths))
        self.length = len(self.paths)

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor()
        ])


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = self.transform(img)
        return img
    
class normalize(object):
    def __init__(self):
        self.max_value = 1.0 

    def __call__(self, tensor):
        out = (2 * (tensor.float() / self.max_value) - 1.) * 1024
        return out
    
class PytorchVisionXRayDataset(Dataset):
    def __init__(self, path, resolution=224):
        super().__init__()
        self.folder = path #csv_file_name
        import pandas as pd
        df = pd.read_csv(path)
        self.paths = list(np.asarray(df['lateral_512_jpeg']))
        print("Number of files: ", len(self.paths))
        self.length = len(self.paths)
        
        
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Lambda(normalize()),
            transforms.Lambda(expand_greyscale())
        ])


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        path = self.paths[index]
        sample = Image.open(path)
        img = self.transform(sample)
        return img