import argparse
import math
import random
import os
import sys
import yaml
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from dataset import MultiResolutionDataset
from dataset import AFHQDataset, SkinDataset, CelebADataset, XRayDataset
import pdb
from op import conv2d_gradfix
from styleGANv2 import train, data_sampler
from non_leaking import augment, AdaptiveAugment
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0,"/ocean/projects/asc170022p/nmurali/projects/CounterfactualExplainer/MIMICCX-Chest-Explainer/Classifier/torchxrayvision_")
import torchxrayvision as xrv
from swagan import Generator, Discriminator
sys.path.insert(0,"/ocean/projects/asc170022p/nmurali/projects/CounterfactualExplainer/MIMICCX-Chest-Explainer/Classifier/torchxrayvision_/torchxrayvision")
import models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument(
    '--config', '-c', default='/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Configs/Explainer/styleGAN_AFHQ_ln0p15.yaml')
    parser.add_argument(
    '--main_dir', '-m', default='/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/') 
    args = parser.parse_args()
    main_dir = args.main_dir
    # ============= Load config =============
    config_path = args.config
    config = yaml.safe_load(open(config_path))
    print("Training Configuration: ")
    print(config)
    
    device = config["device"]
    path = os.path.join(args.main_dir, config["path"])
    save_dir = os.path.join(args.main_dir, config["save_dir"])
    cls_ckpt = os.path.join(args.main_dir, config["cls_ckpt"])
    # Setting the seed
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    n_gpu = torch.cuda.device_count()
    print("n_gpu: ",n_gpu)
    config['distributed'] = n_gpu > 1
    if config['distributed']:
        ports = [9101, 9102, 9103, 9104]
        import os
        #os.environ['MASTER_ADDR'] = '10.0.3.29'
        os.environ['MASTER_PORT'] = str(ports[config['local_rank']])
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        #os.environ['NCCL_DEBUG'] = 'INFO'
        #os.environ['GLOO_SOCKET_IFNAME'] = 'enp0s31f6'
        print("local_rank: ",config['local_rank'])
        torch.cuda.set_device(config['local_rank'])
        torch.distributed.init_process_group(backend="nccl",  init_method="env://")
        synchronize()
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except:
        pass
    try:
        if not os.path.exists(os.path.join(save_dir,'encoder_sample')):
            os.makedirs(os.path.join(save_dir,'encoder_sample'))
    except:
        pass
    try:
        if not os.path.exists(os.path.join(save_dir,'checkpoint')):
            os.makedirs(os.path.join(save_dir,'checkpoint'))
    except:
        pass
    
    #Read Dataset
    if config['dataset_name'] == 'AFHQ':
        dataset = AFHQDataset(path,  config['size'])
    elif config['dataset_name'] == 'skin':
        dataset = SkinDataset(path, config['imgpath'], config['size'])
    elif config['dataset_name'] == 'CelebA':
        dataset = CelebADataset(path,  config['size'], config['CenterCrop'])
    elif config['dataset_name'] == 'MIMIC':
        dataset = XRayDataset(path, config['size'])
    else:
        print("Provide an appropriate dataset name")
        sys.exit()
    loader = data.DataLoader(
        dataset,
        batch_size=config['batch'],
        sampler=data_sampler(dataset, shuffle=True, distributed=config['distributed']),
        drop_last=True,
    )

    
    generator = Generator(
        config['size'], config['latent'], config['num_cls'], config['n_mlp'], channel_multiplier=config['channel_multiplier']
    ).to(device)
    discriminator = Discriminator(
        config['size'], concate_size = config['concate_size'], channel_multiplier=config['channel_multiplier']
    ).to(device)

    g_reg_ratio = config['g_reg_every'] / (config['g_reg_every'] + 1)
    d_reg_ratio = config['d_reg_every']  / (config['d_reg_every']  + 1)
    
    g_optim = optim.Adam(
        generator.parameters(),
        lr=config['lr'] * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=config['lr'] * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    
    #read classifier
    classifier = xrv.models.DenseNet(num_classes=config['num_cls'], in_channels=3,return_logit=True, drop_rate = config['cls_drop_rate'], weights = cls_ckpt, **xrv.models.get_densenet_params(config['cls_model_type'])) 
    if config['ckpt'] == '':
        config['ckpt'] = None
    if config['ckpt'] is None:
        all_files = os.listdir(os.path.join(save_dir,'checkpoint'))
        current = -1
        current_name = ''
        for f in all_files:
            if '.pt' in f:
                temp = f.split('.')[0]
                temp = int(temp)
                if temp > current:
                    current = temp
                    current_name = f
        if current_name != '':
            config['ckpt'] = os.path.join( save_dir,'checkpoint', current_name)
            
    if config['ckpt'] is not None:
        print("load model:", config['ckpt'])

        ckpt_loaded = torch.load(config['ckpt'], map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(config['ckpt'])
            config['start_iter'] = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt_loaded["g"])
        discriminator.load_state_dict(ckpt_loaded["d"])
        g_optim.load_state_dict(ckpt_loaded["g_optim"])
        d_optim.load_state_dict(ckpt_loaded["d_optim"])
    if torch.cuda.is_available():
            classifier.cuda()
    if config['distributed']:        
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[config['local_rank']],
            output_device=config['local_rank'],
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[config['local_rank']],
            output_device=config['local_rank'],
            broadcast_buffers=False,
        )
        
        classifier = nn.parallel.DistributedDataParallel(
            classifier,
            device_ids=[config['local_rank']],
            output_device=config['local_rank'],
            broadcast_buffers=False,
        )

    

    train(config, loader, generator, discriminator, g_optim, d_optim, classifier, device, save_dir = save_dir)
