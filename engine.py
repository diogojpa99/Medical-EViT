# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for adjusting keep rate and visualization -- Youwei Liang
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from helpers import adjust_keep_rate
from visualize_mask import get_real_idx, mask, save_img_batch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score, \
    roc_curve, auc, roc_auc_score

import numpy as np
import matplotlib.pyplot as plt

def train_one_epoch(model: torch.nn.Module, 
                criterion: torch.nn.Module,
                data_loader: Iterable, 
                optimizer: torch.optim.Optimizer,
                device: torch.device, 
                epoch: int, 
                loss_scaler, 
                max_norm: float = 0,
                lr_scheduler=None,
                model_ema: Optional[ModelEma] = None,
                set_training_mode=True,
                wandb=print,
                args=None):
    
    # Put model in train mode
    model.train(set_training_mode)

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    train_stats = {}
    preds = []; targs = []

    # Evit Parameters
    ITERS_PER_EPOCH = len(data_loader)
    base_rate = args.base_keep_rate
    lr_num_updates = it = epoch * len(data_loader)

    # Loop through data loader data batches
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        
        # Send data to target device
        inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
        
        # Compute keep rate
        keep_rate = adjust_keep_rate(it, epoch, warmup_epochs=args.shrink_start_epoch,
                                         total_epochs=args.shrink_start_epoch + args.shrink_epochs,
                                         ITERS_PER_EPOCH=ITERS_PER_EPOCH, base_keep_rate=base_rate)

        # 1. Clear gradients
        optimizer.zero_grad()

        # 2. Forward pass
        with torch.cuda.amp.autocast():
            scores = model(inputs, keep_rate)
            loss = criterion(scores, labels)
            
        train_loss += loss.item() 
        if not math.isfinite(train_loss):
            print("Loss is {}, stopping training".format(train_loss))
            sys.exit(1)

        if loss_scaler is not None:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward() # 3. Backward pass
            
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) # 4. Clip gradients
                
            optimizer.step() # 5. Update weights

        # Update LR Scheduler
        if not args.cosine_one_cycle:
            lr_scheduler.step_update(num_updates=lr_num_updates)
        
        # Update Model Ema
        if model_ema is not None:
            if device == 'cuda:0' or device == 'cuda:1':
                torch.cuda.synchronize()
            model_ema.update(model)

        # Calculate and accumulate accuracy metric across all batches
        predictions = torch.argmax(torch.softmax(scores, dim=1), dim=1)
        train_acc += (predictions == labels).sum().item()/len(scores)

        preds.append(predictions.cpu().numpy()); targs.append(labels.cpu().numpy())
        
        it += 1
        
        #left_tokens = model.left_tokens
        train_stats['keep_rate'] = keep_rate
        train_stats['left_tokens'] = model.left_tokens


    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(data_loader)
    train_acc = train_acc / len(data_loader)

    train_stats['train_loss'] = train_loss
    train_stats['train_acc'] = train_acc
    train_stats['train_lr'] = optimizer.param_groups[0]['lr']
    
    if wandb != print:
        wandb.log({"Keep Rate":keep_rate}, step=epoch)
        wandb.log({"Train Loss":train_loss} ,step=epoch)
        wandb.log({"Train Accuracy":train_acc}, step=epoch)
        wandb.log({"Train LR":optimizer.param_groups[0]['lr']}, step=epoch)
        
    # Compute Metrics
    preds=np.concatenate(preds); targs=np.concatenate(targs)
    train_stats['confusion_matrix'], train_stats['f1_score'] = confusion_matrix(targs, preds), f1_score(targs, preds, average=None) 
    train_stats['precision'], train_stats['recall'] = precision_score(targs, preds, average=None), recall_score(targs, preds, average=None)
    train_stats['bacc'] = balanced_accuracy_score(targs, preds)
    train_stats['acc1'], train_stats['loss'] = train_acc, train_loss
    
    return train_stats, keep_rate

@torch.no_grad()
def evaluate(model: torch.nn.Module, 
            dataloader: torch.utils.data.DataLoader, 
            keep_rate: None,
            criterion: torch.nn.Module, 
            device: torch.device,
            epoch: int,
            wandb=print,
            args=None):
    
    # Switch to evaluation mode
    model.eval()
    
    preds = []
    targets = []
    test_loss, test_acc = 0, 0
    results = {}
    
    for inputs, targets_ in dataloader:
        
        inputs, targets_ = inputs.to(device, non_blocking=True), targets_.to(device, non_blocking=True)

        # Compute output
        with torch.cuda.amp.autocast():
            scores = model(inputs, keep_rate)
            loss = criterion(scores, targets_)
        
        test_loss += loss.item()
    
        # Calculate and accumulate accuracy
        predictions = scores.argmax(dim=1)
        test_acc += ((predictions == targets_).sum().item()/len(predictions))
        
        preds.append(predictions.cpu().numpy())
        targets.append(targets_.cpu().numpy())

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)

    if wandb!=print:
        wandb.log({"Val Loss":test_loss},step=epoch)
        wandb.log({"Val Accuracy":test_acc},step=epoch)
        
    # Compute Metrics
    preds=np.concatenate(preds); targets=np.concatenate(targets)
    results['confusion_matrix'], results['f1_score'] = confusion_matrix(targets, preds), f1_score(targets, preds, average=None) 
    results['precision'], results['recall'] = precision_score(targets, preds, average=None), recall_score(targets, preds, average=None)
    results['bacc'] = balanced_accuracy_score(targets, preds)
    results['acc1'], results['loss'] = accuracy_score(targets, preds), test_loss

    return results

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score + self.delta:
            # If we don't have an improvement, increase the counter 
            self.counter += 1
            #self.trace_func(f'\tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # If we have an imporvement, save the model
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        

@torch.no_grad()
def get_acc(data_loader, model, device, keep_rate=None, tokens=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, keep_rate, tokens)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return metric_logger.acc1.global_avg


@torch.no_grad()
def visualize_mask(data_loader, model, device, output_dir, n_visualization, fuse_token, keep_rate=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualize:'
    rank = 0
    world_size = 0
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(3, 1, 1)

    # switch to evaluation mode
    model.eval()

    ii = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        B = images.size(0)

        with torch.cuda.amp.autocast():
            output, idx = model(images, keep_rate, get_idx=True)
            loss = criterion(output, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        # denormalize
        images = images * std + mean

        idxs = get_real_idx(idx, fuse_token)
        for jj, idx in enumerate(idxs):
            masked_img = mask(images, patch_size=16, idx=idx)
            save_img_batch(masked_img, output_dir, file_name='img_{}' + f'_l{jj}.jpg', start_idx=world_size * B * ii + rank * B)

        save_img_batch(images, output_dir, file_name='img_{}_a.jpg', start_idx=world_size * B * ii + rank * B)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.synchronize_between_processes()
        ii += 1
        if world_size * B * ii >= n_visualization:
            break

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
