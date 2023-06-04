import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os 

import visualize_mask


def get_predicted_class(image_file, predicted_class):
    if image_file[0].lower() == 'm':
        prefix = 'Mel'
    elif image_file[0].lower() == 'n':
        prefix = 'NV'
    else:
        prefix = ''
    
    pred_class = 'MEL' if predicted_class == 0 else 'NV'
    
    return f'{prefix} | Pred: {pred_class}'

def Grad_CAM(input, model, keep_rate, prediction, label, img):    
    
    prediction[:, label].backward(retain_graph=True)
    gradients = model.get_activations_gradient()
    gradients = gradients[:,1:]
    gradients = gradients.permute(0,2,1)
    gradients = gradients.reshape(1,384,14,14)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(input,keep_rate)[0].detach()
    activations = activations.unsqueeze(0)
    activations = activations[:,1:]
    activations = activations.permute(0,2,1)
    activations = activations.reshape(1,384,14,14)
    
    
    for i in range(256):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap) 

    heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), scale_factor=(224//14), mode='bilinear',align_corners=True)  #14->224
    heatmap = heatmap.reshape(224, 224).data.cpu().numpy() 
    heatmap = cv2.applyColorMap(np.uint8(heatmap* 255), cv2.COLORMAP_JET)   
   
    heatmap = np.float32(heatmap) / 255
    cam = heatmap*0.9 + np.float32(img)
    cam = cam / np.max(cam)
    
    vis =  np.uint8(255 * cam)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        
    return vis

def visualize_masks_aux(model:torch.nn.Module, datapath, keep_rate, fuse_token, device, outputdir = None, agrs = None):
    
    image_files = os.listdir(datapath)
    fig, axs = plt.subplots(2, len(image_files), figsize=(4*len(image_files), 12))
    
    # Transform the images for the model
    transform = transforms.Compose(
    [   transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(3, 1, 1)
    
    for i, image_file in enumerate(image_files):
        
        image_path = os.path.join(datapath, image_file)
        image = Image.open(image_path)
        
        input = transform(image).unsqueeze(0)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get the output of the model
        output, idx = model(input, keep_rate, get_idx=True)
        predicted_class = int(torch.argmax(output))
        num_left_patches = model.left_tokens
        
        # Denormalize img
        img = input * std + mean
        
        # Evit Mask
        idxs = visualize_mask.get_real_idx(idx, fuse_token)
        masked_img = visualize_mask.mask(img, patch_size=16, idx=idxs[-1]) # Visualize the mask for the last layer
                
        # Plot the original image
        axs[0, i].imshow(masked_img.squeeze().permute(1, 2, 0).data.cpu().numpy())
        axs[0, i].set_title(get_predicted_class(image_file, predicted_class), fontsize=16)
        axs[0, i].axis('off');
        
    image_files = os.listdir(datapath)        
    for i, image_file in enumerate(image_files):
        
        image_path = os.path.join(datapath, image_file)
        image = Image.open(image_path)
        
        input = transform(image).unsqueeze(0)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get the output of the model
        output = model(input, 1)
        predicted_class = int(torch.argmax(output))
        num_left_patches_1 = model.left_tokens
        
        # Denormalize img
        #img = input * std + mean
        img = input.permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))   
                
        # Grad CAM
        grad_cam = Grad_CAM(input,model, 1, output,0, img) # See this better
        
        # Plot the heatmap overlay
        """ axs[1, i].imshow(vis)
        axs[1, i].set_title("Last Layer Attention Map")
        axs[1, i].axis('off'); """
        
        # Plot the original 14x14 heatmap
        axs[1, i].imshow(grad_cam)
        axs[1, i].set_title("Grad-CAM")
        axs[1, i].axis('off');
                    
    title = f"| EViT Mask Visualization | Keep Rate: {keep_rate} | Patches throught the layers: {num_left_patches} and {num_left_patches_1}|"
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(str(outputdir) + f'/EVIT_Mask-Keep_Rate_{keep_rate}-Left_Patches_{list(num_left_patches)[-1]}-Heatmap_01-MEL.jpg', dpi=300, bbox_inches='tight')

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

def generate_attention_map(model,tensor_image,head_fusion="mean",discard_ratio=0.9,method="last_layer_attn"):
    
    output = model(tensor_image,1)
    label=torch.argmax(output,1)
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32) 
    one_hot[0, label] = 1 
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)  
    one_hot = torch.sum(one_hot * output) 

    model.zero_grad()  
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
            attn_cams.append(attention_heads_fused) #avg of the heads  b,n,n 
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
    
    transformer_attribution = generate_attention_map(model,original_image,
                                                     method=method,head_fusion=head_fusion,
                                                     discard_ratio=discard_ratio).detach()


    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    #print(transformer_attribution.shape)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear',align_corners=True)  #14->224
    #print(transformer_attribution.shape)
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()  #/(224,224)
    #print(transformer_attribution.shape)
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())


    image_transformer_attribution = original_image.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    #print(image_transformer_attribution.shape)
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    
    return vis

def visualize_masks(model:torch.nn.Module, datapath, keep_rate, fuse_token, device, outputdir = None, agrs = None):
    
    image_files = os.listdir(datapath)
    fig, axs = plt.subplots(4, len(image_files), figsize=(4*len(image_files), 20))
    
    # Transform the images for the model
    transform = transforms.Compose(
    [   transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(3, 1, 1)
    
    for i, image_file in enumerate(image_files):
        
        image_path = os.path.join(datapath, image_file)
        image = Image.open(image_path)
        
        input = transform(image).unsqueeze(0)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get the output of the model
        output, idx = model(input, keep_rate, get_idx=True)
        predicted_class = int(torch.argmax(output))
        num_left_patches = model.left_tokens
        
        # Denormalize img
        img = input * std + mean
        
        # Evit Mask
        idxs = visualize_mask.get_real_idx(idx, fuse_token)
        masked_img = visualize_mask.mask(img, patch_size=16, idx=idxs[-1]) # Visualize the mask for the last layer
        
        last_layer_attn = generate_visualization(input,model,method="last_layer_attn")
        Grad_CAM = generate_visualization(input,model,method="Grad_Rollout_last_layer")
                
        # Plot the original image
        axs[0, i].imshow(image)
        axs[0, i].set_title(get_predicted_class(image_file, predicted_class), fontsize=16)
        axs[0, i].axis('off');
        
        # Plot Evit Mask
        axs[1, i].imshow(masked_img.squeeze().permute(1, 2, 0).data.cpu().numpy())
        axs[1, i].set_title("EVIT Mask")
        axs[1, i].axis('off');
        
        # Plot attention map from the last layer
        axs[2, i].imshow(last_layer_attn)
        axs[2, i].set_title("Last Layer Attention Map")
        axs[2, i].axis('off');
        
        # Plot grad cam
        axs[3, i].imshow(Grad_CAM)
        axs[3, i].set_title("Grad-CAM")
        axs[3, i].axis('off');
        

    title = f"| EViT Mask Visualization | Keep Rate: {keep_rate} | Patches throught the layers: {num_left_patches} |"
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(str(outputdir) + f'/EVIT_Mask-Keep_Rate_{keep_rate}-Left_Patches_{list(num_left_patches)[-1]}-Predicted_Class.jpg', dpi=300, bbox_inches='tight')        
