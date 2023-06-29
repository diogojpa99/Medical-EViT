# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Impemented by Diogo Araújo
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from pathlib import Path
from typing import List, Union

from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma


from datasets import build_dataset
from engine import train_one_epoch, evaluate, visualize_mask, EarlyStopping
from losses import DistillationLoss, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from samplers import RASampler
import utils, models, visualization
from helpers import speed_test, get_macs

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

import wandb

def get_args_parser():
    
    parser = argparse.ArgumentParser('EViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--gpu', default='cuda:1', help='GPU id to use.')
    
    # EViT parameters
    parser.add_argument('--test_speed', action='store_true', help='whether to measure throughput of model')
    parser.add_argument('--only_test_speed', action='store_true', help='only measure throughput of model')
    
    parser.add_argument('--fuse_token', action='store_true', help='whether to fuse the inattentive tokens')
    parser.add_argument('--base_keep_rate', type=float, default=0.7,
                        help='Base keep rate (default: 0.7)')
    
    parser.add_argument('--shrink_epochs', default=0, type=int, 
                        help='how many epochs to perform gradual shrinking of inattentive tokens')
    parser.add_argument('--shrink_start_epoch', default=10, type=int, 
                        help='on which epoch to start shrinking of inattentive tokens')
    
    parser.add_argument('--drop_loc', default='(3, 6, 9)', type=str, 
                        help='the layer indices for shrinking inattentive tokens')
    
    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str, help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'], type=str, help='Image Net dataset path')
    parser.add_argument('--use-lmdb', action='store_true', help='use Image Net lmdb dataset (for data-set==IMNET)')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    
    parser.add_argument('--pretrained_dataset_name', default='ImageNet1k', type=str, help='pretrained dataset name')
    parser.add_argument('--finetune_dataset_name', default='ISIC2019-Clean', type=str, choices=['ISIC2019-Clean', 'PH2', 'Derm7pt'], metavar='DATASET')

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
   
    parser.add_argument('--eval', action='store_true', default=False, help='Perform evaluation only') 
    parser.add_argument('--finetune_flag', action='store_true', default=True, help='Perform Finetuning and eval')
    parser.add_argument('--train_flag', action='store_true', default=False, help='Train from scratch and eval') 

    ## Visualize mask
    parser.add_argument('--visualize_mask', action='store_true', help='Visualize the dropped image patches and then exit')
    parser.add_argument('--n_visualization', default=128, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',  help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',help='')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Wanb parameters
    parser.add_argument('--project_name', default='Thesis', help='name of the project')
    parser.add_argument('--hardware', default='Server', choices=['Server', 'Colab', 'MyPC'], help='hardware used')
    parser.add_argument('--run_name', default='MIL', help='name of the run')
    parser.add_argument('--wandb', action='store_false', default=True, help='whether to use wandb')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')

    # Imbalanced dataset parameters
    parser.add_argument('--class_weights', action='store_true', default=True, help='Enabling class weighting')
    parser.add_argument('--class_weights_type', default='Manual', choices=['Median', 'Manual'], type=str, help="")
    
    # Optimizer parameters 
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', choices=['adamw', 'sgd'],
                        help='Optimizer (default: "adamw")')
    
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')

    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters 
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', choices=['step', 'multistep', 'cosine', 'plateau','poly', 'exp'],
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    
    # * Lr Cosine Scheduler Parameters
    parser.add_argument('--cosine_one_cycle', type=bool, default=False, help='Only use cosine one cycle lr scheduler')
    parser.add_argument('--lr_k_decay', type=float, default=1.0, help='LR k rate (default: 1.0)')
    parser.add_argument('--lr_cycle_mul', type=float, default=1.0, help='LR cycle mul (default: 1.0)')
    parser.add_argument('--lr_cycle_decay', type=float, default=1.0, help='LR cycle decay (default: 1.0)')
    parser.add_argument('--lr_cycle_limit', type=int, default=1, help= 'LR cycle limit(default: 1)')
    
    parser.add_argument('--lr-noise', type=Union[float, List[float]], default=None, help='Add noise to lr')
    parser.add_argument('--lr-noise-pct', type=float, default=0.1, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.1)')
    parser.add_argument('--lr-noise-std', type=float, default=0.05, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 0.05)')
    
    # * Warmup parameters
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_lr', type=float, default=1e-3, metavar='LR',
                        help='warmup learning rate (default: 1e-3)')
    
    parser.add_argument('--min_lr', type=float, default=1e-4, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')

    # * StepLR parameters
    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    
    # * MultiStepLRScheduler parameters
    parser.add_argument('--decay_milestones', type=List[int], nargs='+', default=(10, 15), 
                        help='epochs at which to decay learning rate')
    
    # * The decay rate is transversal to many schedulers | However it has a different meaning for each scheduler
    # MultiStepLR: decay factor of learning rate | PolynomialLR: power factor | ExpLR: decay factor of learning rate
    parser.add_argument('--decay_rate', '--dr', type=float, default=1., metavar='RATE', help='LR decay rate (default: 0.1)')

    # Model EMA parameters -> Exponential Moving Average Model
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=12, metavar='N')
    parser.add_argument('--delta', type=float, default=0.0, metavar='N')
    parser.add_argument('--counter_saver_threshold', type=int, default=12, metavar='N')
    
    # Data augmentation parameters 
    parser.add_argument('--batch_aug', action='store_true', default=False, help='whether to augment batch')
    parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT', help='Color jitter factor (default: 0.)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + \
                        "(default: rand-m9-mstd0.5-inc1)'),
    
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.1, metavar='PCT', help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='const', help='Random erase mode (default: "const")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')
    
    # Loss scaler
    parser.add_argument('--loss_scaler', action='store_true', default=False, help='Use loss scaler')
    
    # New Visualization Params
    parser.add_argument('--visualize_complete', action='store_true', help='Visualize evit mask, last layer attention and grad_cam.')
    parser.add_argument('--visualize_num_images', default=8, type=int, help="")
    parser.add_argument('--vis_num', default=1, type=int, help="")
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--images_path', type=str, default='', help='Path to the images')

    return parser


def main(args):
    
    # Start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            project=args.project_name,
            config={
            "model": args.model, "Pretrained Dataset": args.pretrained_dataset_name, "Finetune Dataset": args.finetune_dataset_name,
            "epochs": args.epochs,"batch_size": args.batch_size, "Finetune": args.finetune_flag, "Eval": args.eval,
            "warmup_epochs": args.warmup_epochs, "Warmup lr": args.warmup_lr,
            "cooldown_epochs": args.cooldown_epochs, "patience_epochs": args.patience_epochs,
            "lr_scheduler": args.sched, "lr": args.lr, "min_lr": args.min_lr,
            "dropout": args.drop, "weight_decay": args.weight_decay,
            "optimizer": args.opt, "momentum": args.momentum,
            "seed": args.seed, "class_weights": args.class_weights,
            "early_stopping_patience": args.patience, "early_stopping_delta": args.delta,
            "model_ema": args.model_ema, "Batch_augmentation": args.batch_aug, "Loss_scaler": args.loss_scaler,
            "PC": args.hardware,
            }
        )
        wandb.run.name = args.run_name
        
        if args.debug:
            wandb=print
    
    if args.train_flag or args.finetune_flag:
        print('-------------------------------------------')
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print('-------------------------------------------')

    # Set device
    device = args.gpu if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    
    ################## Data Setup ##################
    
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val,_ = build_dataset(is_train=False, args=args)
        
    ##### Data Loaders 
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
        
    ##################### Create model  ########################
    
    print(f"Creating model: {args.model}"); print(f"Base_keep_rate: {args.base_keep_rate}"); print(f"drop_loc: {eval(args.drop_loc)}")
    print(f"num_classes: {args.nb_classes}"); print(f"drop_rate: {args.drop}"); print(f"fuse_token: {args.fuse_token}")
    print(f"img_size: ({args.input_size, args.input_size})\n")
    
    model = create_model(
        args.model,
        base_keep_rate=args.base_keep_rate,
        drop_loc=eval(args.drop_loc),
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        fuse_token=args.fuse_token,
        img_size=(args.input_size, args.input_size)
    )
    
    ## Load the pretrained model 
    if args.finetune:
        utils.Load_Pretrained_Model_Finetuning(model, args)

    # Set model to device   
    model.to(device)

    # Set output directory
    output_dir = Path(args.output_dir)

    ##############  Test speed ##############
    if (args.test_speed or args.only_test_speed) and utils.is_main_process():
        # test model throughput for three times to ensure accuracy
        inference_speed = speed_test(model)
        print('inference_speed (inaccurate):', inference_speed, 'images/s')
        inference_speed = speed_test(model)
        print('inference_speed:', inference_speed, 'images/s')
        inference_speed = speed_test(model)
        print('inference_speed:', inference_speed, 'images/s')
        MACs = get_macs(model)
        print('GMACs:', MACs * 1e-9)

        def log_func1(*arg, **kwargs):
            log1 = ' '.join([f'{xx}' for xx in arg])
            log2 = ' '.join([f'{key}: {v}' for key, v in kwargs.items()])
            log = log1 + "\n" + log2
            log = log.strip('\n') + '\n'
            if args.output_dir and utils.is_main_process():
                with (output_dir / "speed_macs.txt").open("a") as f:
                    f.write(log)
        log_func1(inference_speed=inference_speed, GMACs=MACs * 1e-9)
        log_func1(args=args)
        
        if args.only_test_speed:
            return
    
    ### Model EMA 
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    ##################################  PARAMETERS #####################################
    
    # Class wighting
    class_weights = utils.Class_Weighting(dataset_train, dataset_val, device, args)
    
    # Number of parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_parameters}\n")
    if args.test_speed and utils.is_main_process():
        log_func1(n_parameters=n_parameters * 1e-6)
        
    # Create optimizer
    optimizer = create_optimizer(args,model)
    
    # Define the loss scaler
    loss_scaler = NativeScaler() if args.loss_scaler else None
    
    # Create scheduler
    if args.lr_scheduler:
        if args.sched == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
        else:    
            lr_scheduler,_ = create_scheduler(args, optimizer)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    if utils.is_main_process():
        print("output_dir:", args.output_dir)

    ######################################### Resume Model  ###############################################
    
    if args.resume:
        utils.Load_Pretrained_Model(model, optimizer, lr_scheduler, loss_scaler, model_ema, args)
    
        if args.visualize_mask:
            print('******* Starting visualization process. *******')
            visualize_mask(data_loader_val, model, device, args.output_dir, args.n_visualization, args.fuse_token)
            return
    
        if args.visualize_complete:
            print('******* Starting visualization process. *******')
            val_loader = visualization.VisualizationLoader_Binary(dataset_val, args)
            #visualization.Visualize_Activation(model=model, dataloader=val_loader, device=device, keep_rate=args.base_keep_rate, outputdir=args.output_dir,args=args)
            visualization.Visualize_Activation_Rollout(model=model, dataloader=val_loader, device=device, keep_rate=args.base_keep_rate, outputdir=args.output_dir, args=args)
            return

        if args.eval:
            print('******* Starting evaluation process. *******')
            best_results = evaluate(model=model, 
                                    dataloader=data_loader_val, 
                                    keep_rate= args.base_keep_rate,
                                    criterion= torch.nn.CrossEntropyLoss(), 
                                    device= device  ,
                                    epoch = 0,
                                    args=args)
        
    if args.finetune_flag or args.train_flag:

        start_time = time.time()
        train_results = {'loss': [], 'acc': [] , 'lr': [], 'left_tokens':[]}
        val_results = {'loss': [], 'acc': [], 'f1': [], 'cf_matrix': [], 'bacc': [], 'precision': [], 'recall': []}
        best_val_bacc = 0.0
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=args.delta, path=str(output_dir) +'/checkpoint.pth')
        
        print(f"******* Start training for {(args.epochs + args.cooldown_epochs)} epochs. *******") 
        for epoch in range(args.start_epoch, (args.epochs + args.cooldown_epochs)):

            train_stats, keep_rate = train_one_epoch(model = model, 
                                                     criterion = criterion, 
                                                     data_loader =data_loader_train,
                                                     optimizer = optimizer,
                                                     device = device,
                                                     epoch=(epoch+1),
                                                     loss_scaler=loss_scaler,
                                                     max_norm=args.clip_grad,
                                                     lr_scheduler=lr_scheduler,
                                                     model_ema=model_ema,
                                                     set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
                                                     wandb=wandb,
                                                     args=args)
            
            # Learning rate scheduler step
            if args.lr_scheduler:
                lr_scheduler.step((epoch+1))
            
            results = evaluate(model=model,
                                dataloader=data_loader_val,
                                keep_rate=keep_rate,
                                criterion=criterion,
                                device=device,
                                epoch=(epoch+1),
                                wandb=wandb,
                                args=args)   
            
            # Update results dictionary
            train_results['loss'].append(train_stats['train_loss']); train_results['acc'].append(train_stats['train_acc']); train_results['lr'].append(train_stats['train_lr'])
            train_results['left_tokens'].append(train_stats['left_tokens'])
            val_results['acc'].append(results['acc1']); val_results['loss'].append(results['loss']); val_results['f1'].append(results['f1_score'])
            val_results['cf_matrix'].append(results['confusion_matrix']); val_results['precision'].append(results['precision'])
            val_results['recall'].append(results['recall']); val_results['bacc'].append(results['bacc'])

            print(f"Epoch: {epoch+1} | lr: {train_stats['train_lr']:.5f} | Train Loss: {train_stats['train_loss']:.4f} | Train Acc: {train_stats['train_acc']:.4f} |",
                  f"Val. Loss: {results['loss']:.4f} | Val. Acc: {results['acc1']:.4f} | Val. Bacc: {results['bacc']:.4f} | F1-score: {np.mean(results['f1_score']):.4f}")
            
            if results['bacc'] > best_val_bacc and early_stopping.counter < args.counter_saver_threshold:
                # Only want to save the best checkpoints if the best val bacc and the early stopping counter is less than the threshold
                best_val_bacc = results['bacc']
                checkpoint_paths = [output_dir / f'EViT-DropLoc_{args.drop_loc}-KeepRate_{args.base_keep_rate}-best_checkpoint.pth']
                best_results = results
                for checkpoint_path in checkpoint_paths:
                    checkpoint_dict = {
                        'model':model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.lr_scheduler:
                        checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
                    if model_ema is not None:
                        checkpoint_dict['model_ema'] = get_state_dict(model_ema)
                    utils.save_on_master(checkpoint_dict, checkpoint_path)
                print(f"\tBest Val. Bacc: {(best_val_bacc*100):.2f}% |[INFO] Saving model as 'best_checkpoint.pth'")
                                
            # Early stopping
            early_stopping(results['loss'], model)
            if early_stopping.early_stop:
                print("\t[INFO] Early stopping - Stop training")
                break
            
        # Compute the total training time
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))        
        print('\n---------------- Train stats for the last epoch ----------------\n',
            f"Acc: {train_stats['acc1']:.3f} | Bacc: {train_stats['bacc']:.3f} | F1-score: {np.mean(train_stats['f1_score']):.3f} | \n",
            f"Precision[MEL]: {train_stats['precision'][0]:.3f} | Precision[NV]: {train_stats['precision'][1]:.3f} | \n",
            f"Recall[MEL]: {train_stats['recall'][0]:.3f} | Recall[NV]: {train_stats['recall'][1]:.3f} | \n",
            f"Confusion Matrix: {train_stats['confusion_matrix']}\n",
            f"Training time {total_time_str}\n",
            f"Number of attentive tokens throught the Encoder blocs: {train_stats['left_tokens']}\n")
        
        utils.plot_loss_and_acc_curves(train_results, val_results, output_dir=output_dir, args=args)
    
    utils.plot_confusion_matrix(best_results["confusion_matrix"], {'MEL': 0, 'NV': 1}, output_dir=output_dir, args=args)
    
    print('\n---------------- Val. stats for the best model ----------------\n',
        f"Acc: {best_results['acc1']:.3f} | Bacc: {best_results['bacc']:.3f} | F1-score: {np.mean(best_results['f1_score']):.3f} | \n",
        f"Precision[MEL]: {best_results['precision'][0]:.3f} | Precision[NV]: {best_results['precision'][1]:.3f} | \n",
        f"Recall[MEL]: {best_results['recall'][0]:.3f} | Recall[NV]: {best_results['recall'][1]:.3f} | \n")
    
    if wandb != print:
        wandb.log({"Best Val. Acc": best_results['acc1'], "Best Val. Bacc": best_results['bacc'], "Best Val. F1-score": np.mean(best_results['f1_score'])})
        wandb.log({"Best Val. Precision[MEL]": best_results['precision'][0], "Best Val. Precision[NV]": best_results['precision'][1]})
        wandb.log({"Best Val. Recall[MEL]": best_results['recall'][0], "Best Val. Recall[NV]": best_results['recall'][1]})
        wandb.log({"Training time": total_time})
        #wandb.finish()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
