import numpy as np
import sys
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image as PILImage
sys.path.insert(0,'/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/utils')
from my_datasets import *
sys.path.insert(0,'/jet/home/nmurali/asc170022p/nmurali/projects/augmentation_by_explanation_eccv22/Misc/energy_ood/utils')
from display_results import *


parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# User params
parser.add_argument('--test_name', default='mnist', type=str)
parser.add_argument('--chkpt_path', default='/jet/home/nmurali/asc170022p/singla/CounterfactualExplainer/MIMICCX-Chest-Explainer/Classifier/torchxrayvision_/Experiment_Simulation_MNIST/classifier_64_Few_Classes_Dirty_MNIST_seed_1234/MNIST-densenet121-MNIST_64-best.pt', type=str)
parser.add_argument('--ood_type', default='food2', type=str, help='aid|food1|food2')
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
args = parser.parse_args()
print(args)


# load dataset
_, _, test_loader = get_dataloader(args.test_name, 1234)
ood_loader = get_ood_loaders(args.test_name, args.ood_type)

# Create model
# net = get_model(args.test_name)
# net.load_state_dict(torch.load(args.chkpt_path))
net = torch.load(args.chkpt_path)
net.eval()
net.cuda()

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_loader)*test_loader.batch_size // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_loader)*test_loader.batch_size)

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, samples in enumerate(tqdm(loader)):
            try:
                data = samples['img']
            except:
                try:
                    data = samples[0]
                except:
                    raise('Unable to iterate loader!')
            if batch_idx>500:
                break
            # if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            #     break

            data = data.cuda()

            output = net(data)
            # smax = to_np(F.softmax(output, dim=1))

            _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))

            # if in_dist:
            #     preds = np.argmax(smax, axis=1)
            #     targets = target.numpy().squeeze()
            #     right_indices = preds == targets
            #     wrong_indices = np.invert(right_indices)

            #     _right_score.append(-np.max(smax[right_indices], axis=1))
            #     _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

in_score = get_ood_scores(test_loader, in_dist=True)


# /////////////// End Detection Prelims ///////////////

# /////////////// Error Detection ///////////////

# print('\n\nError Detection')
# show_performance(wrong_score, right_score, method_name=args.test_name)

# /////////////// OOD Detection ///////////////

def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)

auroc_list, aupr_list, fpr_list = [], [], []

get_and_print_results(ood_loader)


# /////////////// Mean Results ///////////////

print('\n\nMean Validation Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
