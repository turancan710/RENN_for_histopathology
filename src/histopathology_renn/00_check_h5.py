import h5py
#import os
from PIL import Image
import numpy as np
#from histo_feat_extr_backbone import split_tile_input_backbone 
import torchvision.transforms as T
import torch

h5_path = "./tiles/decbdda7-e62a-4436-b233-28c5353d0f61.h5"
file_id = "decbdda7-e62a-4436-b233-28c5353d0f61"

inv3_transform = T.Compose([
    T.Resize((299,299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],#pretrained on imagenet data (=imagenet mean&std dev)
                std=[0.229, 0.224, 0.225])
    ])

with h5py.File(h5_path, 'r') as f:
    tile_stack = f[file_id]['tiles']
    print(f'stack of tiles shape: {tile_stack.shape}')
    print(f'stack of tiles dtype: {tile_stack.dtype}')

    '''
    for i in range(tile_stack.shape[0]):
            tile = tile_stack[i]
            print(f'Tile {i}: shape = {tile.shape} , dtype= {tile.dtype}')
    '''
    img_np = tile_stack[5]
    img = Image.fromarray(img_np.astype(np.uint8))
    print(img.mode)
    img.save("debug_image.png")

    '''#below check dimensions following transformations of patches
    patches_np = split_tile_input_backbone(img_np)
    
    patch_batch = []
    for patch in patches_np:
        patch_img = Image.fromarray(patch.astype(np.uint8))
        print(patch_img.mode)
        patch_tensor = inv3_transform(patch_img).unsqueeze(0)  #1,3,299,299
        print(patch_tensor.shape)
        patch_batch.append(patch_tensor)
    
    patch_batch = torch.stack(patch_batch)
    print(patch_batch.shape)
    '''

    