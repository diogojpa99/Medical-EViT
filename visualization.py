import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import DataLoader, Subset, ConcatDataset

import visualize_mask


def fix_map(map, idxs, N, device):
    
    full_map = torch.zeros((N)).to(device)
    full_map = full_map + map
    
    return full_map
    
def Cam_Select_Attn(attn:torch.Tensor):
    """ 
    Select the first row of the attention map (the one that corresponds to the CLS token).
    Then compute the mean of all the attention heads.
    """
    
    cls_attn = attn[:, :, 0, 1:]
    #cam = cam.clamp(min=0).mean(dim=1) # Mean of the heads
    cls_attn = cls_attn.mean(dim=1)
    
    return cls_attn.squeeze(0)
    
    
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
        
    return joint_attention

def Gen_Attn_Map(output,
                 label,
                 model,
                 idxs,
                 method="last_layer_attn",
                 head_fusion="mean",
                 discard_ratio=0.9,
                 device:torch.device=None):

    model.zero_grad()  
    output[:,label].backward(retain_graph=True)
    
    if method == "rollout":
        attn_cams = []
        for block in model.blocks:
            attn_heads = block.attn.get_attn().clamp(min=0)
            if head_fusion == "mean":
              attention_heads_fused = attn_heads.mean(axis=1) #1,heads, n_tokens,n_tokens 
            elif head_fusion == "max":
                attention_heads_fused = attn_heads.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attn_heads.min(axis=1)[0]  
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)  #1,(n_tokens*n_tokens) | flat.shape -> 1, (n_tokens*n_tokens)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False) #discard 
            indices = indices[indices != 0]
            flat[0, indices] = 0
            attn_cams.append(attention_heads_fused) #avg of the heads  b,n,n 
        cam = compute_rollout_attention(attn_cams)
        cam = cam[:, 0, 1:] 
    
    elif method == "Grad_Rollout":
        cams = []
        for block in model.blocks:
            grad = block.attn.get_attn_gradients().squeeze(0)
            cam = block.attn.get_attn().squeeze(0)
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams)
        cam = rollout[:, 0, 1:]
    
    elif method == "Grad_Rollout_last_layer":
        grad = model.blocks[-1].attn.get_attn_gradients()
        cam = model.blocks[-1].attn.get_attn()
        cam = grad * cam
        cam = Cam_Select_Attn(cam)
        
    elif method == "last_layer_cam":
        cam = model.blocks[-1].attn.get_attn()
        cam = Cam_Select_Attn(cam)

    
    elif method == "middle_layer_attn":
        cam = model.blocks[5].attn.get_attn()
        cam = Cam_Select_Attn(cam)


    cam = fix_map(cam, model.num_tokens, device)
    return cam

def GenVis(output,
           label,
           image, 
           model, 
           idxs,
           method="rollout", 
           head_fusion = "mean", 
           discard_ratio = 0.9,
           device:torch.device=None):
    
    transformer_attribution = Gen_Attn_Map(output,
                                           label,
                                           model,
                                           idxs,
                                           method,
                                           head_fusion,
                                           discard_ratio,
                                           device).detach()

    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    vis = ShowVis(transformer_attribution, image)
    return vis


#### Utils Functions

def ShowVis(activation_map, img):
    
    heatmap = torch.nn.functional.interpolate(activation_map, scale_factor=(224//14), mode='bilinear', align_corners=True)  #14->224
    heatmap = heatmap.reshape(224, 224).data.cpu().numpy() 
    heatmap = cv2.applyColorMap(np.uint8(heatmap* 255), cv2.COLORMAP_JET)        
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap*0.9 + np.float32(img)
    cam = cam / np.max(cam)
    
    vis =  np.uint8(255 * cam)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    return vis

def Collate_Binary(batch):
    
    # (1) Separate the batch into MEL (0) and NV (1) classes
    mel = [item for item in batch if item[1]==0]
    nv = [item for item in batch if item[1]==1]
    
    # (2) Determine the desired number of instances for each class in the batch
    instances_per_class = len(batch) // 2
    
    # Slice the batches to have an equal number of instances for each class
    mel = mel[:instances_per_class]
    nv = nv[:instances_per_class]
    
    return mel+nv
  
def VisualizationLoader_Binary(val_set:torch.utils.data.Dataset, args=None):
    
    # (1) Obtain the idxs of the melanoma and nevus samples    
    mel_idx=[]; nv_idx=[]
    for i, (_, label,_ ,_) in enumerate(val_set):
        if label==0: 
            mel_idx.append(i)
        elif label==1: 
            nv_idx.append(i)
        if i==len(val_set)-1:
            break

    # (2) Shuffle the indices randomly
    """ random.shuffle(mel_idx)
    random.shuffle(nv_idx) """

    # Select an equal number of indices for each class
    num_samples_per_class = min(len(mel_idx), len(nv_idx))
    mel_idx = mel_idx[:num_samples_per_class]
    nv_idx = nv_idx[:num_samples_per_class]
    
    # (3) Create Subset objects for each class
    mel_subset = Subset(val_set, mel_idx)
    nv_subset = Subset(val_set, nv_idx)
    
    # (4) Create separate DataLoaders for each class subset
    mel_loader = DataLoader(mel_subset, batch_size= (args.visualize_num_images//2), shuffle=True, collate_fn=Collate_Binary)
    nv_loader = DataLoader(nv_subset, batch_size=(args.visualize_num_images//2), shuffle=True, collate_fn=Collate_Binary)
    
    return DataLoader(ConcatDataset([mel_loader.dataset, nv_loader.dataset]), batch_size=args.visualize_num_images, shuffle=True)

def Get_Predicted_Class(label, predicted_class):
    if label == 0:
        prefix = 'Mel'
    elif label == 1:
        prefix = 'NV'
    else:
        prefix = ''
    
    pred_class = 'MEL' if predicted_class == 0 else 'NV'
    
    return f'{prefix} | Pred: {pred_class}'
        
#### Visualize Activation Maps #####

def Visualize_Activation(model: torch.nn.Module, 
                        dataloader:torch.utils.data.DataLoader, 
                        device:torch.device,
                        keep_rate:float=None,
                        outputdir=None, 
                        args=None):

    fig, axs = plt.subplots(4, args.visualize_num_images, figsize=(4*(args.visualize_num_images), 17))
    
    mean = IMAGENET_DEFAULT_MEAN; std = IMAGENET_DEFAULT_STD
    reverse_transform = transforms.Compose([
        transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]), (1.0 / std[0], 1.0 / std[1], 1.0 / std[2])),
        transforms.ToPILImage()
    ])

    for j, (inputs, labels) in enumerate(dataloader):
        
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        for i in range(args.visualize_num_images):
            
            input=inputs[i].unsqueeze(0)
            image=reverse_transform(inputs[i])
                            
            # (3) Set model to eval mode
            model.eval()
            
            # (4) Obtain The Output of the Model
            with torch.cuda.amp.autocast():
                output, idx = model(input, keep_rate, get_idx=True)
            predicted_class = int(torch.argmax(output))

            # (7) Normalize the input image
            img = input.permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
            img = (img - np.min(img)) / (np.max(img) - np.min(img))    

            # (9) Obtain the class probabilities for 'MEL' heatmap 
            #rollout=GenVis(output=output,label=0,image=img,model=model, idxs=idx, head_fusion="mean",discard_ratio=0,device=device)
            """ rollout_discard=GenVis(output=output,label=0,image=img,model=model, idxs=idx, head_fusion="mean",device=device)
            rollout_max=GenVis(output=output,label=0,image=img,model=model, idxs=idx, head_fusion="max",device=device)
            rollout_min=GenVis(output=output,label=0,image=img,model=model, idxs=idx, head_fusion="min",device=device) """
            #Grad_rollout=GenVis(output=output,label=0,image=img,model=model, idxs=idx, method="Grad_Rollout",device=device)
            Grad_rollout_last=GenVis(output=output,label=0,image=img,model=model, idxs=idx, method="Grad_Rollout_last_layer",device=device)
            rollout_last_layer=GenVis(output=output,label=0,image=img,model=model, idxs=idx, method="last_layer_attn",device=device)
            #rollout_middle_layer=GenVis(output=output,label=0,image=img,model=model, idxs=idx, method="middle_layer_attn",device=device)
            
            # Plot the original image
            axs[0, i].imshow(image)
            axs[0, i].set_title(Get_Predicted_Class(labels[i], predicted_class), fontsize=16)
            axs[0, i].axis('off');

            # Plot Probability Heatmap
            axs[1, i].imshow(Grad_rollout_last)
            axs[1, i].set_title("'MEL' Probability Heatmap")
            axs[1, i].axis('off');
            
            # Plot Grad-CAM
            axs[2, i].imshow(rollout_last_layer)
            axs[2, i].set_title("Grad-CAM [MEL]")
            axs[2, i].axis('off');
                       
        title = f"| MIL Class Activation Maps ({args.dataset}) | MIL Type: {args.mil_type} | Pooling Type: {args.pooling_type} |"
        plt.suptitle(title, fontsize=20)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(str(outputdir) + f'/MIL-{args.mil_type}-{args.pooling_type}-{args.dataset}-Class_Activations-sBatch_{j}.jpg', dpi=300, bbox_inches='tight')  
        
        if j == (args.vis_num-1):
            break   
