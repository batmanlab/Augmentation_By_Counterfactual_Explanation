import os, sys
import pickle
import pprint
import random
from glob import glob
from os.path import exists, join
from torch.autograd import Variable
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn, sklearn.model_selection
import datasets
from torch.utils.tensorboard import SummaryWriter
from random import random as rand
import wandb
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)
#from tqdm.auto import tqdm


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def adjust_learning_rate(cfg, optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = cfg.lr
    if epoch in [10, 15,20,30]:
        print("Old lr: ", lr)
        lr /= 10
        print("New lr: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def uniform_binning(y_conf,bin_size=0.10):
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    lower_bounds = upper_bounds- bin_size
    y_bin = []
    n_bins = len(upper_bounds)
    for s in y_conf:
        if (s <= upper_bounds[0]) :
            y_bin.append(0)
        elif (s > lower_bounds[n_bins-1]) :
            y_bin.append(n_bins-1)
        else:
            for i in range(1,n_bins-1):
                if (s > lower_bounds[i]) & (s <=upper_bounds[i] ):
                    y_bin.append(i)
                    break
    y_bin = np.asarray(y_bin)
    return y_bin

def train(model, dataset, cfg, train_inds=np.empty([]), test_inds=np.empty([])):        
    dataset_name = cfg['dataset'] + "-" + cfg['model'] + "-" + cfg['name']
    
    device = 'cuda' if cfg['cuda'] else 'cpu'
    if not torch.cuda.is_available() and cfg['cuda']:
        device = 'cpu'
        print("WARNING: cuda was requested but is not available, using cpu instead.")

    print(f'Using device: {device}')
    print("Saving everying at path:")
    print(cfg['output_dir'])

    if not exists(cfg['output_dir']):
        os.makedirs(cfg['output_dir'])
    
    if SummaryWriter:
        log_dir = os.path.join(cfg['output_dir'], 'log')
        if not exists(log_dir):
            os.makedirs(log_dir)
        logger = SummaryWriter(log_dir) 

    # Dataset 
    if dataset is not None:
        if len(train_inds.shape) == 0:
            gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.85,test_size=0.15, random_state=cfg['seed'])
            train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patient_id))
            train_dataset = datasets.SubsetDataset(dataset, train_inds)
            valid_dataset = datasets.SubsetDataset(dataset, test_inds)
            np.save(os.path.join(cfg['output_dir'], 'train.npy'), train_inds)
            np.save(os.path.join(cfg['output_dir'], 'validation.npy'), test_inds)
        else:
            train_dataset = datasets.SubsetDataset(dataset, train_inds)
            valid_dataset = datasets.SubsetDataset(dataset, test_inds)
        print("Train: ", train_dataset)
        print("Validation: ", valid_dataset)
    else:
        train_dataset = train_inds
        valid_dataset = test_inds
    
    
    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg['batch_size'],
                                               shuffle=cfg['shuffle'],
                                               num_workers=cfg['threads'], 
                                               pin_memory=cfg['cuda'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cfg['batch_size'],
                                               shuffle=cfg['shuffle'],
                                               num_workers=cfg['threads'], 
                                               pin_memory=cfg['cuda'])

    # Optimizer
    if cfg['opt'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5, amsgrad=True)
    elif cfg['opt'] == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-5)
    print("optimizer: ", optim)
    
    
    criterion = torch.nn.BCEWithLogitsLoss()

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    weights_files = glob(join(cfg['output_dir'], f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(cfg['output_dir'], f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        model.load_state_dict(torch.load(weights_file).state_dict())

        with open(join(cfg['output_dir'], f'{dataset_name}-e{start_epoch}-metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        best_metric = metrics[-1]['best_metric']
        weights_for_best_validauc = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    model.to(device)
    
    for epoch in range(start_epoch, cfg['num_epochs']):
        avg_loss = train_epoch(cfg=cfg,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               criterion=criterion,
                               logger=logger)
            
        auc_valid, task_aucs, task_outputs, task_targets = valid_test_epoch(cfg=cfg, 
                                     name='Valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterion=criterion,
                                     logger=logger)
        
        if np.mean(auc_valid) > best_metric:
            try:
                os.remove(join(cfg['output_dir'], f'{dataset_name}-best-auc{best_metric:4.4f}.pt')) # remove previous best checkpoint
            except:
                pass
            best_metric = np.mean(auc_valid)
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(cfg['output_dir'], f'{dataset_name}-best-auc{np.mean(auc_valid):4.4f}.pt'))
            # only compute when we need to

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric,
            'task_aucs': task_aucs[0],
            'task_recall':task_aucs[1],
        }

        metrics.append(stat)

        with open(join(cfg['output_dir'], f'{dataset_name}-e{epoch + 1}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        if epoch%cfg['save_freq']==0:
            torch.save(model, join(cfg['output_dir'], f'{dataset_name}-e{epoch + 1}-auc{np.mean(auc_valid):4.4f}.pt'))   
        
        #adjust_learning_rate(cfg, optim, epoch)

    return metrics, best_metric, weights_for_best_validauc
      
def train_epoch(cfg, epoch, model, device, train_loader, optimizer, criterion, limit=None, logger=None):
    model.train()

    if cfg['taskweights']:
        weights = np.nansum(train_loader.dataset.labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights/weights.max()
        weights = torch.from_numpy(weights).to(device).float()
        print("task weights", weights)
    
    avg_loss = []
    t = tqdm(train_loader)
    num_batches = len(train_loader)

    for batch_idx, samples in enumerate(t):
        if limit and (batch_idx > limit):
            print("breaking out")
            break
            
        optimizer.zero_grad()
        try:
            images = samples["img"].float().to(device)
            targets = samples["lab"].to(device)
        except:
            images = samples[0].float().to(device)
            targets = samples[1].to(device)

        if len(targets.shape) == 1 and cfg['num_classes'] != 1:
            targets = F.one_hot(targets, num_classes=cfg['num_classes'])

        if cfg['mixUp']:
            images, targets_a, targets_b, lam = mixup_data(images, targets,
                                                       cfg['alpha'])
            images, targets_a, targets_b = map(Variable, (images,
                                                      targets_a, targets_b))
        if cfg['labelSmoothing']:
            targets = targets *(1 - cfg['alpha']) + cfg['alpha']/targets.shape[0]
        outputs = model(images)
        loss = torch.zeros(1).to(device).float()
        for task in range(targets.shape[1]):
            task_output = outputs[:,task]
            task_target = targets[:,task]
            if len(task_target) > 0:
                if not (cfg['mixUp'] or cfg['focalLoss']):
                    if len(cfg['pos_weights'])>0:
                        wt = torch.tensor([cfg['pos_weights'][task]]).to('cuda')
                        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=wt)
                        task_loss = criterion(task_output.float(), task_target.float())
                    else:
                        task_loss = criterion(task_output.float(), task_target.float())
                    if  torch.isnan(task_loss):
                        import pdb
                        pdb.set_trace()
                elif cfg['focalLoss']:
                    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                    task_loss = criterion(task_output.float(), task_target.float())
                    pt = torch.exp(-task_loss) # prevents nans when probability 0
                    focal_loss = 0.25 * (1-pt)**cfg['alpha'] * task_loss
                    task_loss =  focal_loss.mean()
                elif cfg['mixUp']:
                    task_targets_a =targets_a[:,task]
                    task_targets_b =targets_b[:,task]
                    task_loss = mixup_criterion(criterion, task_output.float(), task_targets_a.float(), task_targets_b.float(), lam)
                if cfg['taskweights']:
                    loss += weights[task]*task_loss
                else:
                    loss += task_loss

        index = epoch*len(t)+batch_idx
        logger.add_scalar('train/task_loss', loss, index)
        logger.add_scalar('train/lr',optimizer.param_groups[0]['lr'], index)
                
        loss = loss.sum()
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()
        logger.add_scalar('train/total_loss', loss, index)
        if cfg['featurereg']:
            feat = model.features(images)
            loss += feat.abs().sum()
            
        if cfg['weightreg']:
            loss += model.classifier.weight.abs().sum()
        
        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

        optimizer.step()

        if cfg['save_iters']>0 and epoch%cfg['save_freq']==0:
            idx_flag = int(num_batches/cfg['save_iters'])
            if batch_idx%idx_flag==0:
                torch.save(model, join(cfg['output_dir'], f'e{epoch + 1}-it{batch_idx}.pt'))   
    return np.mean(avg_loss)

def valid_test_epoch(cfg, name, epoch, model, device, data_loader, criterion, limit=None, logger =None):
    model.eval()

    avg_loss = []
    task_outputs=[]
    task_targets=[]
    for task in range(cfg['num_classes']):
        task_outputs.append( [])
        task_targets.append( [])
        
    with torch.no_grad():
        t = tqdm(data_loader)
        
        # iterate dataloader
        for batch_idx, samples in enumerate(t):
            index = epoch*len(t)+batch_idx
            if limit and (batch_idx > limit):
                print("breaking out")
                break
            try:
                images = samples["img"].to(device)
                targets = samples["lab"].to(device)
            except:
                images = samples[0].to(device)
                targets = samples[1].to(device)
            if len(targets.shape) == 1 and cfg['num_classes'] != 1:
                targets = F.one_hot(targets, num_classes=cfg['num_classes'])
            outputs = model(images)
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                if len(task_target) > 0:
                    if len(cfg['pos_weights'])>0:
                        wt = torch.tensor([cfg['pos_weights'][task]]).to('cuda')
                        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=wt)
                        loss += criterion(task_output.double(), task_target.double())
                    else:
                        loss += criterion(task_output.double(), task_target.double())
                    if torch.isnan(loss):
                        import pdb
                        pdb.set_trace()
                task_output = torch.sigmoid(task_output)
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            try:
                logger.add_scalar('valid/total_loss', loss, index)
            except:
                pass
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            logger.add_scalar('valid/avg_loss', np.mean(avg_loss), index)
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
          
        task_aucs = []
        task_recalls = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                try:
                    task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                except:
                    continue
                task_aucs.append(task_auc)
                y_pred = np.asarray(task_outputs[task] > 0.5).astype(int)
                task_recall = sklearn.metrics.recall_score(task_targets[task], y_pred)
                task_recalls.append(task_recall)
                
                cm = sklearn.metrics.confusion_matrix(task_targets[task], y_pred) 
                print(cm, task, 0.5, "epoch: ", epoch)
                
                if not cfg['multi_class']: #its multi-label, calculate ece for each task
                    task_conf = (task_outputs[task] * y_pred) + (1-y_pred) * (1-task_outputs[task])
                    task_bin = uniform_binning(task_outputs[task],bin_size=0.10)
                    n_bins = np.max(task_bin)+1
                    N = task_bin.shape[0]
                    ece = 0
                    for b in range(n_bins):
                        index = np.where(task_bin == b)
                        y_pred_b = y_pred[index]
                        y_true_b = task_targets[task][index]
                        y_conf_b = task_conf[index]
                        if y_pred_b.shape[0] == 0:
                            ece +=0 
                        else:
                            acc = np.sum(y_pred_b == y_true_b)/y_pred_b.shape[0]
                            c = y_pred_b.shape[0]
                            conf = np.mean(y_conf_b)
                            ece +=  np.abs(acc - conf) * (c/N)
                    logger.add_scalar('valid/ece_task_'+str(task), ece, epoch)
                    print('ece task ' +str(task)+' : ', ece)
            else:
                task_aucs.append(np.nan)
                task_recalls.append(np.nan)
        if cfg['multi_class']:
            task_outputs = np.asarray(task_outputs)
            task_targets = np.asarray(task_targets)
            task_outputs = np.transpose(np.asarray(task_outputs))
            task_targets = np.transpose(np.asarray(task_targets))
            y_pred = np.argmax(task_outputs,axis=1)
            y_conf = np.max(task_outputs,axis=1)
            y_true = np.argmax(task_targets,axis=1)
            y_bin = uniform_binning(y_conf,bin_size=0.10)
            n_bins = np.max(y_bin)+1
            N = y_bin.shape[0]
            ece = 0
            for b in range(n_bins):
                index = np.where(y_bin == b)
                y_pred_b = y_pred[index]
                y_true_b = y_true[index]
                y_conf_b = y_conf[index]
                if y_pred_b.shape[0] == 0:
                    ece +=0 
                else:
                    acc = np.sum(y_pred_b == y_true_b)/y_pred_b.shape[0]
                    c = y_pred_b.shape[0]
                    conf = np.mean(y_conf_b)
                    ece +=  np.abs(acc - conf) * (c/N)
            logger.add_scalar('valid/ece', ece, epoch)
            print('ece: ', ece)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    task_recalls = np.asarray(task_recalls)
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')
    print("Tasks AUC:")
    print(task_aucs)
    print("Tasks Recall:")
    print(task_recalls)
    return auc, [task_aucs,task_recalls], task_outputs, task_targets
