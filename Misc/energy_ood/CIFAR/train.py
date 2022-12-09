# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0,'/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/utils')
from my_datasets import *



if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.tinyimages_80mn_loader import TinyImages
    from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='skin')
parser.add_argument('--model', '-m', type=str, default='densenet',
                    choices=['allconv', 'wrn', 'densenet'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/CIFAR/snapshots/pretrained/', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--score', type=str, default='energy', help='OE|energy')
parser.add_argument('--seed', type=int, default=99, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
args = parser.parse_args()


if args.score == 'OE':
    save_info = 'oe_tune'
elif args.score == 'energy':
    save_info = 'energy_ft'

args.save = args.save+save_info
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

# ============= Seed ====================    
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# load dataset
train_loader_in, train_loader_out, val_loader = get_dataloader(args.dataset, args.seed)

# Create model
net = get_model(args.dataset)

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 0
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module
# Restore model
model_found = False

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////

def train():
    net.train()  # enter train mode
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    out_key = 'img'
    if args.dataset=='afhq' or args.dataset=='mnist':
        out_key = 0
    for b_idx, (in_set, out_set) in enumerate(zip(tqdm(train_loader_in), train_loader_out)):
        data = torch.cat((in_set['img'], out_set[out_key]), 0)
        target = in_set['lab']

        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # backward
        scheduler.step()
        optimizer.zero_grad()

        loss = F.cross_entropy(x[:len(in_set['img'])], target)
        # cross-entropy from softmax distribution to uniform distribution
        if args.score == 'energy':
            Ec_out = -torch.logsumexp(x[len(in_set['img']):], dim=1)
            Ec_in = -torch.logsumexp(x[:len(in_set['img'])], dim=1)
            loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())
        elif args.score == 'OE':
            loss += 0.5 * -(x[len(in_set['img']):].mean(1) - torch.logsumexp(x[len(in_set['img']):], dim=1)).mean()

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # if b_idx==50:
        #     break

    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for b_idx, samples in enumerate(tqdm(val_loader)):
            data = samples['img']
            target = samples['lab']
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            correct += output.max(1)[1].eq(target.max(1)[1]).sum().item()
            num_samples += target.shape[0]

            # test loss average
            loss_avg += float(loss.data)

            # if b_idx==50:
            #     break

    state['test_loss'] = loss_avg / num_samples
    state['test_accuracy'] = correct / num_samples


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + '_' + args.model + '_s' + str(args.seed) +
                                  '_' + save_info+'_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_accuracy(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in tqdm(range(0, args.epochs)):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()
 
    # Save model
    if epoch%5==0:
        torch.save(net.state_dict(),
                os.path.join(args.save, args.dataset, args.dataset + '_' + args.model + '_s' + str(args.seed) +
                                '_' + save_info + '_epoch_' + str(epoch) + '.pt'))
    

    # Show results
    with open(os.path.join(args.save, args.dataset + '_' + args.model + '_s' + str(args.seed) +
                                      '_' + save_info + '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Accuracy {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100. * state['test_accuracy'])
    )
