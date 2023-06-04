# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for adjusting keep rate, logging to tensorboard, and
# speed/throughput measurement -- Youwei Liang
import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from torchvision import transforms


import cv2
import evit

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

""" from datasets import build_dataset
from engine import train_one_epoch, evaluate, visualize_mask
from losses import DistillationLoss
from samplers import RASampler
import models
import utils
from helpers import speed_test, get_macs """

from PIL import Image

import matplotlib.pyplot as plt

#from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
        
    return joint_attention

def generate_attention_map(model,tensor_image,head_fusion="mean",discard_ratio=0.9,method="rollout"):
    
    output = model(tensor_image)
    #emb = model(tensor_image)
    #output = F.linear(emb, F.normalize(model.loss.weight))
    label=torch.argmax(output,1)
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32) 
    one_hot[0, label] = 1  #one hot encoding
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)  
    one_hot = torch.sum(one_hot * output) #multiply one hot encoding by the output

    model.zero_grad()  #reset gradients
    one_hot.backward(retain_graph=True)
    
    if method == "rollout":
        attn_cams = []
        for block in model.blocks:
            attn_heads = block.attn.get_attn().clamp(min=0)
            #print(attn_heads.shape)#b,h,n,n  ,only positive values
            if head_fusion == "mean":
              attention_heads_fused = attn_heads.mean(axis=1) #1,heads, n_tokens,n_tokens 
            elif head_fusion == "max":
                attention_heads_fused = attn_heads.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attn_heads.min(axis=1)[0]
                    
                   
               
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)  #1,(n_tokens*n_tokens)
            # flat.shape -> 1, (n_tokens*n_tokens)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False) #discard 
            indices = indices[indices != 0]
            flat[0, indices] = 0
        
                
            #avg of the heads  b,n,n 
            attn_cams.append(attention_heads_fused)

        cam = compute_rollout_attention(attn_cams)
        cam = cam[:, 0, 1:]
        return cam   
    
    elif method == "Grad_Rollout":
            cams = []
            for block in model.blocks:
                grad = block.attn.get_attn_gradients()
                cam = block.attn.get_attn()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams)
            cam = rollout[:, 0, 1:]
            return cam     
    
    elif method == "Grad_Rollout_last_layer":
        grad = model.blocks[-1].attn.get_attn_gradients()
        cam = model.blocks[-1].attn.get_attn()
        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        cam = cam[0, 1:]
        return cam  
        
    elif method == "last_layer_attn":
        cam = model.blocks[-1].attn.get_attn()
        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=0)
        cam = cam[0, 1:]
        return cam   
    
    elif method == "middle_layer_attn":
        cam = model.blocks[5].attn.get_attn()
        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=0)
        cam = cam[0, 1:]
        return cam

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization(original_image,model,method="rollout", head_fusion = "mean", discard_ratio = 0.9):
    
    # Imagem de input: 224x224
    # patch size: 16x16
    # nº de tokens: (224/16)^2 = 14^2
    
    transformer_attribution = generate_attention_map(model,original_image.unsqueeze(0),
                                                     method=method,head_fusion=head_fusion,
                                                     discard_ratio=discard_ratio).detach()


    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    #print(transformer_attribution.shape)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear',align_corners=True)  #14->224
    #print(transformer_attribution.shape)
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()  #/(224,224)
    #print(transformer_attribution.shape)
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())


    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    #print(image_transformer_attribution.shape)
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    
    return vis

def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = evit.EViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=evit.partial(torch.nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = evit._cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

def vit_tiny(pretrained=True, **kwargs):
    
    model = evit.EViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=evit.partial(torch.nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = evit._cfg()
    
    if pretrained:
        model.load_state_dict(torch.load("evit-0.7-fuse-img224-deit-s.pth",map_location="cpu"))
    
    return 


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # arguments related to the shrinking of inattentive tokens
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

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
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
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

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
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--use-lmdb', action='store_true', help='use Image Net lmdb dataset (for data-set==IMNET)')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--visualize_mask', action='store_true', help='Visualize the dropped image patches and then exit')
    parser.add_argument('--n_visualization', default=128, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser




#### main() ####

def main(args):

    #utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)
        
    print(f"Creating model: {args.model}")
    '''model = create_model(
        args.model,
        base_keep_rate=args.base_keep_rate,
        drop_loc=eval(args.drop_loc),
        pretrained=False,
        num_classes=2,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        fuse_token=args.fuse_token,
        img_size=(args.input_size, args.input_size)
    )'''
    
    #model = evit.EViT()
    
    #model = deit_tiny_patch16_224(pretrained=True)

    #model =vit_tiny(True)
    model = evit.deit_small_patch16_224(pretrained=True)
    #checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location='cpu', check_hash=True)

    #checkpoint = torch.load(args.finetune, map_location='cpu')

    '''checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)'''
    
    
    # Data tranform
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    """ image = Image.open("ISIC_0000013.jpg")
    tensor = transform(image)
    #convert_tensor = transforms.ToTensor()

    #tensor=convert_tensor(image)
    print(tensor.shape)
    
    rollout=generate_visualization(tensor,model,head_fusion="mean",discard_ratio=0)
    
    cv2.imwrite("rollout.jpg", rollout) """
    
    fig, axs = plt.subplots(4, 9,figsize=(20, 10))
    
    images = ['a','b','c','d']
    
    for i in range(4):
        
        image = Image.open("imgs_test/" + images[i] + ".jpeg")
    
        tensor = transform(image)
        
        rollout=generate_visualization(tensor,model,head_fusion="mean",discard_ratio=0)
        rollout_discard=generate_visualization(tensor,model,head_fusion="mean")
        rollout_max=generate_visualization(tensor,model,head_fusion="max")
        rollout_min=generate_visualization(tensor,model,head_fusion="min")
        Grad_rollout=generate_visualization(tensor,model,method="Grad_Rollout")
        Grad_rollout_last=generate_visualization(tensor,model,method="Grad_Rollout_last_layer")
        rollout_last_layer=generate_visualization(tensor,model,method="last_layer_attn")
        rollout_middle_layer=generate_visualization(tensor,model,method="middle_layer_attn")

        axs[i][0].imshow(image);
        axs[i][0].title.set_text('INPUT')
        axs[i][0].axis('off');


        axs[i][1].imshow(rollout);
        axs[i][1].title.set_text('rollout tool')
        axs[i][1].axis('off');

        axs[i][2].imshow(rollout_discard);
        axs[i][2].title.set_text('rollout tool 0.9 discard')
        axs[i][2].axis('off');

        axs[i][3].imshow(rollout_last_layer);
        axs[i][3].title.set_text('rollout Last layer')
        axs[i][3].axis('off');

        axs[i][4].imshow(rollout_middle_layer);
        axs[i][4].title.set_text('rollout_mid_layer')
        axs[i][4].axis('off');

        axs[i][5].imshow(rollout_max);
        axs[i][5].title.set_text("rollout_max")
        axs[i][5].axis('off');

        axs[i][6].imshow(rollout_min);
        axs[i][6].title.set_text("rollout_min")
        axs[i][6].axis('off');

        axs[i][7].imshow(Grad_rollout);
        axs[i][7].title.set_text('Grad_rollout')
        axs[i][7].axis('off');

        axs[i][8].imshow(Grad_rollout_last);
        axs[i][8].title.set_text('Grad_rollout_last_layer')
        axs[i][8].axis('off');

    fig.savefig('full_figure2.png')

    for name, module in model.named_modules(): #find the modules
                print(name,module)
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    
    # Aux
    #rollout=generate_visualization(imagem,model,head_fusion="mean",discard_ratio=0)
    #imagem tensor of size 3,224,224