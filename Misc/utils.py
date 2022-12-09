# imports
import torch
import pandas as pd
import sys, yaml
from random import randint as randint
from random import random as rand
from tqdm import tqdm
import torch, torchvision, torchvision.transforms
sys.path.insert(0, '/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Classifier/')
import datasets, models


class Store():
    """
    Store -> get_df -> get_metrics form one pipeline
    as you iterate through data loader, store your values
    use get_df to convert that into a suitable outcomes df
    feed this df into get_metrics to see all metrics on your classifier
        
    use this to store (concatenate) your predictions or labels or ... during training
    it maintains a list of variables (lov): can be cuda tensors or lists
    keep feeding it list of one dimensional tensors (squeezing automatically done), or lists
    and it'll keep concatenating them to existing list
    """
    
    def __init__(self):
        self.lov = [] 
        self.start_flag = True
        
    def __call__(self, mini_lov):
        
        if self.start_flag:
            self.start_flag = False

            for var in mini_lov:
                if isinstance(var, list):
                    self.lov.append(var)
                elif torch.is_tensor(var):
                    if len(var.shape)>1:
                        var = var.squeeze()
                    self.lov.append(var)
                else:
                    raise ValueError('Invalid input type for class "Store"!')
                    
        else:
            for i in range(len(mini_lov)):
                if isinstance(mini_lov[i], list):
                    self.lov[i] += mini_lov[i]

                elif torch.is_tensor(mini_lov[i]):
                    tensor1 = mini_lov[i]            
                    if len(tensor1.shape)>1:
                        tensor1 = tensor1.squeeze()
                    
                    tensor2 = self.lov[i]
                    self.lov[i] = torch.cat((tensor2,tensor1))

                else:
                    raise ValueError('Invalid input type for class "Store"!')
    

def get_df(lov, cols):
    """
    pass the Stores lov into this to convert it into suitable df
    that can be passed to get_metrics    
    """
    df = pd.DataFrame()
    
    for idx, col in enumerate(cols):
        if isinstance(lov[idx],list):
            df[col] = lov[idx]
        elif torch.is_tensor(lov[idx]):
            df[col] = lov[idx].detach().cpu()
        else:
            raise ValueError('Invalid input type for "get_df"!')
        
    return df


def create_data_splits(data_file, column_name_split, new_col, test_id, save_path):

    """
    brief description: 
    * convert train/test splits to train/val/test splits
    * test splits remains same. 15% of train converted to val

    assumptions:
    "column_name_split" field of the "data_file" has 1 for train, 0 for test

    details:
    we create a "new_col" with 1 for train, 0 for val, and "test_id" for test
    """ 

    df = pd.read_csv(data_file)
    df[new_col] = df[column_name_split]

    df_train = df[df[new_col]==1]
    df_test = df[df[new_col]==0]

    df_test[new_col] = test_id

    dfupdate = df_train.sample(frac=0.15)
    dfupdate[new_col] = 0
    df_train.update(dfupdate)
    # update_list = dfupdate.index.tolist() 

    df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
    df_train = df_train.loc[:, ~df_train.columns.str.contains('^Unnamed')]

    df_final = pd.concat([df_train,df_test])

    df_final.to_csv(save_path, index=False)

def corrupt_labels(data_file, class_names, multi_class, noise_level, save_path):
    """
    description: Corrupt the labels of a csv data file with noise.

    assumptions: we assume all columns in data_file corresponding to class_names have 0 or 1 values only!

    info:
    data_file: path to csv file
    class_names: list of class names (pathologies)
    multi_class: (Boolean) True if multi_class, False if multi_label
    noise_level: fraction (0.0 to 1.0)
    save_path: final path including name of csv file to save
    """
    
    ln = noise_level
    all_classes = class_names

    df = pd.read_csv(data_file)
    df = df.sample(frac=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    for i in tqdm(range(int(ln*len(df)))):
        # choose the ith row
        row = df.iloc[i]

        if multi_class:
            # search for a class that is negative in the given row
            while True:
                # randomly sample one of the classes in row
                idx = randint(  0,  len(all_classes)-1  )
                class_name = all_classes[idx]

                # if condition true, make that class positive
                if row[class_name] == 0:
                    row[all_classes] = 0
                    row[class_name] = 1
                    df.iloc[i] = row
                    break
        else:
            row_sum = row[all_classes].sum()

            if row_sum==0:
                row[all_classes] = 1
                df.iloc[i] = row
            else:
                row[all_classes] = 0
                count = 0
                while True:
                    # randomly sample one of the classes in row
                    idx = randint(  0,  len(all_classes)-1  )
                    class_name = all_classes[idx]
                    row[class_name] = 1
                    count+=1
                    if count==row_sum:
                        df.iloc[i] = row
                        break

    df.to_csv(save_path, index=False)

def corrupt_labels_via_cmat(data_file, class_names, s, save_path):
    """
    description: Corrupt labels using a corruption matrix
    refer p-5 of "Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty"
    s: [0,1] corruption strength
    """

    df = pd.read_csv(data_file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    K = len(class_names)
    C = (1-s)*torch.eye(K) + s*torch.ones(K,K)/K

    for i in tqdm(range(len(df))):
        # choose the ith row
        row = df.iloc[i]       
        temp = row[class_names]
        for src_idx,label  in enumerate(row[class_names]):
            if label!=0:
                sum = C[src_idx,0]
                rand_num = rand()
                for targ_idx in range(K):
                    if rand_num<sum:                
                        temp[src_idx] = 0.0
                        temp[targ_idx] = label
                        break
                    else:
                        sum = sum + C[src_idx,targ_idx+1]
        row[class_names] = temp
        df.iloc[i] = row

    df.to_csv(save_path, index=False)


def get_model_preds(model, dataset, data_file, class_names, merge_crit, save_path):
    """
    * run the model over the given dataset and append classifier predictions to the
    data file and corresponding class_names
    * merge_crit is the name of field corresponding to image path/file names
    * final csv is saved to save_path
    """
    model = model.to("cuda")

    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=64,
                                        shuffle=False,
                                        num_workers=4, 
                                        pin_memory=True)

    store = Store()
    
    for b_idx, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            try:
                images = batch["img"].float().to("cuda")
                paths = batch["file_name"]
            except:
                images = batch[0].float().to("cuda")
                paths = batch["file_name"]

            preds = torch.sigmoid(model(images))

            # prepare mini_lov
            mini_lov = [paths]
            for i in range(len(class_names)):
                arr = torch.round(preds[:,i] * 10**4) / (10**4)
                mini_lov.append(arr)

            store(mini_lov)

    new_cols = [  x+'_preds'  for x in class_names  ]
    new_cols = [merge_crit] + new_cols    
    
    df_preds = get_df(store.lov, new_cols)
    df_orig = pd.read_csv(data_file)

    if 'nv' in class_names:
        df_preds['image_id'] = df_preds['image_id'].apply(lambda x: x.split('/')[-1].split('.')[0])

    df_final = pd.merge(df_orig,df_preds,on=merge_crit)
    df_final.to_csv(save_path, index=False)


def get_uniform_preds(data_file, col_name, num_samples):
    """
    The "col_name" in the "data_file" has classifier predictions. 
    Return a data-frame having a max of num_samples in each of 10 bins (from 0 to 1) of the classifier preds
    """
    df = pd.read_csv(data_file)
    df_list= []

    for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        try:
            df_list.append(df[(df[col_name]>=i) & (df[col_name]<i+0.1)].sample(num_samples))
        except:
            df_list.append(df[(df[col_name]>=i) & (df[col_name]<i+0.1)])

    df_final = pd.concat(df_list)

    return df_final

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

def fetch_dataset(config):

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

    elif config['dataset'] == 'MIMIC-CXR':
        transforms = torchvision.transforms.Compose([
            #torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((config['size'], config['size'])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(center_crop()),
            torchvision.transforms.Lambda(normalize())
        ])
        dataset = datasets.MIMIC_Dataset(imgpath=config['imgpath'], csvpath=config['data_file'], class_names=config['class_names'], transform=transforms, seed=config['seed'])

    return dataset