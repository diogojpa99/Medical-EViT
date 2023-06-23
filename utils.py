# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import math
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu) 
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
class CosineCycle:
    def __init__(self, n_splits, min_v, max_v):
        self.n_splits = n_splits
        self.n = 0
        self.values = [min_v + (max_v - min_v) * (math.cos(i / n_splits * 2 * math.pi) + 1) * 0.5
                       for i in range(n_splits)]

    def next(self, update=True, relax=False):
        if relax:
            n = max(0, self.n - 1)
            m = n % self.n_splits
            tmp = self.values[m]
        n = self.n + 1
        m = n % self.n_splits
        if relax and tmp > self.values[m]:
            if update:
                self.n -= 1
            return tmp
        if update:
            self.n += 1
        return self.values[m]

## New

def Load_Pretrained_Model(model, optimizer, lr_scheduler, loss_scaler, model_ema, args):
    
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
        
    #model_without_ddp.load_state_dict(checkpoint['model'])
    model.load_state_dict(checkpoint['model'])
    
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.model_ema:
            _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
            
def Load_Pretrained_Model_Finetuning(model, args):
    
    print("***** Importing model for finetuning. ******\n")
    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.finetune, map_location='cpu')

    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # Interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5) # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5) # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens] # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)
            
def Are_All_Strings_Same(lst):
    if len(lst) == 0:
        print("-> List is empty")
        return True
    first_string = lst[0]
    return all(string == first_string for string in lst)

def Class_Weighting(train_set, val_set, device, args):
    
    # Check the distribution of the dataset
    train_dist = dict(Counter(train_set.targets))
    val_dist = dict(Counter(val_set.targets))
    
    train_dist['MEL'] = train_dist.pop(0)
    train_dist['NV'] = train_dist.pop(1)
    val_dist['MEL'] = val_dist.pop(0)
    val_dist['NV'] = val_dist.pop(1)
    
    n_train_samples = len(train_set)
    
    print(f"Classes: {train_set.classes}\n")
    print(f"Classes map: {train_set.class_to_idx}\n")
    print(f"Train distribution: {train_dist}\n")
    print(f"Val distribution: {val_dist}\n")
    
    if args.class_weights:
        if args.class_weights_type == 'Median':
            class_weight = torch.Tensor([n_train_samples/train_dist['MEL'], 
                                         n_train_samples/ train_dist['NV']]).to(device)
        elif args.class_weights_type == 'Manual':                   
            class_weight = torch.Tensor([n_train_samples/(2*train_dist['MEL']), 
                                         n_train_samples/(2*train_dist['NV'])]).to(device)
    else: 
        class_weight = None
    
    print(f"Class weights: {class_weight}\n")
    
    return class_weight
  
def plot_loss_and_acc_curves(results_train, results_val, output_dir, args):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """
    train_loss = results_train['loss']
    val_loss = results_val['loss']

    train_acc = results_train['acc']
    val_acc = results_val['acc']

    epochs = range(len(results_val['loss']))
    
    """ window_size = 1 # Adjust the window size as needed
    val_loss_smooth = np.convolve(val_loss, np.ones(window_size) / window_size, mode='valid')
    val_acc_smooth = np.convolve(val_acc, np.ones(window_size) / window_size, mode='valid')
    epochs_smooth = range(len(val_loss_smooth)) """
    #plt.figure(figsize=(15, 7))
    fig, axs = plt.subplots(2, 1)

    # Plot the original image
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Val. Loss")
    #axs[0].plot(epochs_smooth, val_loss_smooth, label="Val. Loss")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].legend()
    
    axs[1].plot(epochs, train_acc, label="Train Acc.")
    axs[1].plot(epochs, val_acc, label="Val Acc.")
    #axs[1].plot(epochs_smooth, val_acc_smooth, label="Val. Acc.")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].legend()
    
    plt.subplots_adjust(wspace=2, hspace=0.6)

    # Plot loss
    """ plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs_smooth, val_loss_smooth, label="Val. Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc.")
    plt.plot(epochs_smooth, val_acc_smooth, label="Val. Acc.")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend() """
    
    # Save the figure
    plt.savefig(str(output_dir) + '/loss_curves.png')
    plt.clf()
    
def plot_confusion_matrix(confusion_matrix, class_names, output_dir, args):
    
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(str(output_dir) + '/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.clf()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def plot_loss_curves(results, output_dir):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "test_loss": [...],
            }
    """
    loss = results["train_loss"]
    val_loss = results["test_loss"]
    
    epochs = range(len(results["train_loss"]))
    
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot loss
    ax.plot(epochs, loss, label="train_loss")
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epochs")
    ax.legend()

    # Save the figure
    fig.savefig(str(output_dir) + '/train_test_loss.png')
    
    plt.clf()
    
def plot_roc_curve(fpr, tpr, roc_auc, output_dir):
    
    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
   
    plt.savefig(str(output_dir) + '/roc_curve.png')
    