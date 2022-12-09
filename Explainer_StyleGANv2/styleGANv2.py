import math
import random
import os
import sys
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import utils
from tqdm import tqdm
from dataset import MultiResolutionDataset
from dataset import AFHQDataset, SkinDataset, CelebADataset, XRayDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
import pdb
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment
from torch.utils.tensorboard import SummaryWriter

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def d_hinge_loss(real, fake):
    real_loss = -1*(torch.min(torch.zeros(real.shape[0]).to(device), -1.0 + real[:,0])  ).mean()  #-tf.reduce_mean(tf.minimum(0., -1.0 + real))
    fake_loss = -1*(torch.min(torch.zeros(real.shape[0]).to(device), -1.0 -fake[:,0])  ).mean()  #-tf.reduce_mean(tf.minimum(0., -1.0 - fake))
        
    return real_loss + fake_loss #(F.relu(1 + real) + F.relu(1 - fake)).mean()

def g_hinge_loss(fake):
    return -fake.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(config, loader, generator, discriminator, g_optim, d_optim, classifier, device, save_dir):
    loader = sample_data(loader)
    pbar = range(config['iter'])

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=config['start_iter'], dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    cls_loss = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}
    if SummaryWriter:
        logger = SummaryWriter(save_dir+'checkpoint') 
    if config['distributed']:
        g_module = generator.module
        d_module = discriminator.module
        c_module = classifier.module
    else:
        g_module = generator
        d_module = discriminator
        c_module = classifier

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = config['augment_p'] if config['augment_p'] > 0 else 0.0
    r_t_stat = 0

    if config['augment'] and config['augment_p'] == 0:
        ada_augment = AdaptiveAugment(config['ada_target'], config['ada_length'], 8, device)
        print("ada_augment: ", ada_augment)
    
    #sample_z = torch.randn(config['n_sample'], config['latent'], device=device)
    
    for idx in pbar:
        i = idx + config['start_iter']

        if i > config['iter']:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(classifier, False)
        
        real_pred_cls, real_feature_cls = classifier(real_img)
        real_pred_cls = torch.sigmoid(real_pred_cls)
        
        if config['cls_multi_label'] == False:
            y_t = np.asarray(real_pred_cls.detach().cpu())
            destination_class = np.random.choice(range(config['num_cls']),1)[0]#choose destination class for the current batch
            new_probs = np.random.random(y_t.shape[0])
            max_index = np.argmax(y_t,axis=1)  #current class with max probability
            for ii in range(y_t.shape[0]):
                y_t[ii,max_index[ii]] = 1 - new_probs[ii] #assign a new probability to the currently class
                y_t[ii, destination_class] = new_probs[ii]
            y_t = torch.from_numpy(y_t).float().to(device)
        else:
            y_t = np.asarray(real_pred_cls.detach().cpu())
            if config['destination_cls'] == -1:
                destination_class = np.random.choice(range(config['num_cls']),1)[0]#choose destination class for the current batch
            else:
                destination_class = config['destination_cls']
            y_t[:,destination_class] = np.random.random(y_t.shape[0])
            y_t = torch.from_numpy(y_t).float().to(device)
        
        # Run generator
        recon_x_s, x_s_latent = generator(real_img, real_pred_cls, return_latents=True)#self reconstruction
        fake_x_t, _ = generator(real_img, y_t, return_latents=False) # fake generation
        
        if config['concate_size']!=0:
            x_s_pred = discriminator(real_img, real_feature_cls)
            _, recon_x_s_cls_feature = classifier(recon_x_s)
            recon_x_s_pred = discriminator(recon_x_s, recon_x_s_cls_feature)
            
            fake_y_t, fake_x_t_cls_feature = classifier(fake_x_t)
            fake_y_t = torch.sigmoid(fake_y_t)
            fake_x_t_pred = discriminator(fake_x_t, fake_x_t_cls_feature)
        else:
            x_s_pred = discriminator(real_img, None)
            recon_x_s_pred = discriminator(recon_x_s, None)
            fake_x_t_pred = discriminator(fake_x_t, None)    
        
        d_loss_recon = d_logistic_loss(x_s_pred, recon_x_s_pred) #d_adv
        d_adv_loss_t = d_logistic_loss(x_s_pred, fake_x_t_pred)
        d_loss_total = d_loss_recon  + d_adv_loss_t
         
        loss_dict["d"] = d_loss_total
        loss_dict["recon_real"] = d_loss_recon
        loss_dict["d_adv_loss_t"] = d_adv_loss_t
        loss_dict["real_score"] = x_s_pred.mean()
        loss_dict["recon_score"] = recon_x_s_pred.mean()
        loss_dict["fake_score"] = fake_x_t_pred.mean()
               
        discriminator.zero_grad()
        d_loss_total.backward()
        d_optim.step()

        if config['augment'] and config['augment_p'] == 0: #False
            ada_aug_p = ada_augment.tune(real_pred_cls)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % config['d_reg_every'] == 0

        if d_regularize:
            real_img.requires_grad = True

            if config['augment']:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img
            if config['concate_size']!=0:
                _, real_img_aug_cls_feature = classifier(real_img_aug)
                real_pred = discriminator(real_img_aug, real_img_aug_cls_feature)
            else:
                real_pred = discriminator(real_img_aug, None)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (config['r1'] / 2 * r1_loss * config['d_reg_every'] + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss
        
        #encoder + generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(classifier, False)
        
        # Run generator
        recons_x_s, x_s_latent = generator(real_img, real_pred_cls, return_latents=True)#self reconstruction
        fake_x_t, _ = generator(real_img, y_t, return_latents=False) # fake generation
        fake_x_s, _ = generator(fake_x_t, real_pred_cls, return_latents=False) #cyclic reconstruction
        _, fake_x_s_latent = generator(fake_x_s, real_pred_cls, return_latents=True)
        fake_y_t, fake_x_t_cls_feature = classifier(fake_x_t)
        fake_y_t = torch.sigmoid(fake_y_t)
        fake_y_s, fake_x_s_cls_feature = classifier(fake_x_s)
        fake_y_s = torch.sigmoid(fake_y_s)
        if config['concate_size']!=0:
            _, recons_x_s_cls_feature = classifier(recons_x_s)
            recons_x_s_pred = discriminator(recons_x_s, recons_x_s_cls_feature)          
            fake_x_t_pred = discriminator(fake_x_t, fake_x_t_cls_feature)
        else:
            recons_x_s_pred = discriminator(recons_x_s, None)
            fake_x_t_pred = discriminator(fake_x_t, None)
        
        #GAN loss
        g_loss_recon = g_nonsaturating_loss(recons_x_s_pred) * config['adv']
        g_loss_recon_t = g_nonsaturating_loss(fake_x_t_pred) * config['adv']
        #Reconstruction - loss
        recon_l1_loss = F.l1_loss(recons_x_s, real_img) * config['l1'] 
        recon_l1_cyclic_loss = F.l1_loss(fake_x_s, real_img) * config['l1']
        recon_encoder_l1_loss = F.mse_loss(x_s_latent, fake_x_s_latent) * config['el1']
        
        #KL distance loss between classification outcome
        real_p = y_t + 1.0000e-08 #[:,config['source_task']]
        fake_q = fake_y_t + 1.0000e-08 #[:,config['source_task']]
        cls_KL_loss = (real_p * torch.log(fake_q) ) + ( (1-real_p) * torch.log((1-fake_q)+ 1.0000e-08) )
        cls_KL_loss = -cls_KL_loss.mean() * config['dkl'] 
        
        real_p = real_pred_cls + 1.0000e-08#[:,config['source_task']]
        fake_q = fake_y_s + 1.0000e-08 #[:,config['source_task']]
        cls_KL_loss_s = (real_p * torch.log(fake_q) ) + ( (1-real_p) * torch.log((1-fake_q)+ 1.0000e-08) )
        cls_KL_loss_s = -cls_KL_loss_s.mean() * config['dkl'] 
        
        g_total_loss =  g_loss_recon + g_loss_recon_t+ recon_l1_loss + recon_l1_cyclic_loss + recon_encoder_l1_loss
        if not (torch.isnan(cls_KL_loss) or torch.isinf(cls_KL_loss)):
            g_total_loss = g_total_loss + cls_KL_loss
        if not (torch.isnan(cls_KL_loss_s) or torch.isinf(cls_KL_loss_s)):
            g_total_loss = g_total_loss + cls_KL_loss_s
        
        loss_dict["g"] = g_total_loss
        loss_dict["self-l1"] = recon_l1_loss 
        loss_dict["cyclic-l1"] = recon_l1_cyclic_loss 
        loss_dict["e_l1"] =recon_encoder_l1_loss
        loss_dict["g_recon"] = g_loss_recon
        loss_dict["g_recon_t"] = g_loss_recon_t
        loss_dict["cls_KL_loss"] = cls_KL_loss
        loss_dict["cls_KL_loss_s"] = cls_KL_loss_s
        
        
        generator.zero_grad()
        g_total_loss.backward()
        g_optim.step()

        g_regularize = i % config['g_reg_every'] == 0
        g_regularize = False
        if g_regularize:
            path_batch_size = max(1, config['batch'] // config['path_batch_shrink'])
            noise = mixing_noise(path_batch_size, config['latent'], config['mixing']    , device)
            fake_img, latents = generator(noise, return_latents=True, do_something=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = config['path_regularize'] * config['g_reg_every'] * path_loss

            if config['path_batch_shrink']:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )
        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        d_recon_real = loss_reduced["recon_real"].mean().item()
        d_recon_t = loss_reduced["d_adv_loss_t"].mean().item()
        d_real_score = loss_reduced["real_score"].mean().item()
        d_recon_score = loss_reduced["recon_score"].mean().item()
        d_recon_score_t = loss_reduced["fake_score"].mean().item() 
              
        g_loss_val = loss_reduced["g"].mean().item()
        g_l1 = loss_reduced["self-l1"].mean().item()
        g_cyc_l1 = loss_reduced["cyclic-l1"].mean().item()
        g_el1 = loss_reduced["e_l1"].mean().item()
        g_recon = loss_reduced["g_recon"].mean().item()
        g_recon_t = loss_reduced["g_recon_t"].mean().item()
        g_cls_kl = loss_reduced["cls_KL_loss"].mean().item()
        g_cls_kl_s = loss_reduced["cls_KL_loss_s"].mean().item()
             
        r1_val = loss_reduced["r1"].mean().item()
        
        if SummaryWriter :
            logger.add_scalar('G_loss/total', g_loss_val, i)
            logger.add_scalar('G_loss/l1', g_l1, i)
            logger.add_scalar('G_loss/cycle_l1', g_cyc_l1, i)
            logger.add_scalar('G_loss/e_l1', g_el1, i)
            logger.add_scalar('G_loss/recon', g_recon, i)
            logger.add_scalar('G_loss/recon_t', g_recon_t, i)
            logger.add_scalar('G_loss/cls_kl', g_cls_kl, i)
            logger.add_scalar('G_loss/cls_kl_s', g_cls_kl_s, i)
            
            logger.add_scalar('D_loss/total', d_loss_val, i)     
            logger.add_scalar('D_loss/recon', d_recon_real, i)    
            logger.add_scalar('D_loss/recon_t', d_recon_t, i)    
            logger.add_scalar('D_loss/r1', r1_val, i)    
            logger.add_scalar('D_loss/real_score', d_real_score, i) 
            logger.add_scalar('D_loss/recon_score', d_recon_score, i) 
            logger.add_scalar('D_loss/recon_score_t', d_recon_score_t, i) 
            
        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; d-recon: {d_recon_real:.4f};  d-fake: {d_recon_t:.4f}; g: {g_loss_val:.4f}; g-recon: {g_recon:.4f}; g-fake: {d_recon_t:.4f}; cls_KL: {g_cls_kl:.4f};  l1: {g_l1:.4f}; l1-cycle: {g_cyc_l1:.4f}; e_l1:{g_el1:.4f} "
                    f"augment: {ada_aug_p:.4f}"
                )
            )


            if i % 100 == 0:
                with torch.no_grad():
                    sample = torch.cat([real_img.detach(), recons_x_s.detach(), fake_x_t.detach(), fake_x_s.detach()])
                    utils.save_image(
                        sample,
                        save_dir+f"encoder_sample/{str(i).zfill(6)}.png",
                        nrow=int(config['batch']),
                        normalize=True,
                        range=(0, 1),
                    )
                
            if i % 500 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "config": config,
                        "ada_aug_p": ada_aug_p,
                    },
                    save_dir+f"checkpoint/{str(i).zfill(6)}.pt",
                )

