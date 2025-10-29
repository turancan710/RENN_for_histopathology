import os
import numpy as np
import h5py
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import inception_v3 , Inception_V3_Weights
import torchvision.transforms as T

# === feature extractor (2048-dim feature vector) & requir. transforms ====================================
def get_inception_v3_feature_extractor(device):
    weights_imgnet = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights_imgnet, aux_logits=True) #bug: aux_logits = False = no class head , but code wont run with pretrained weights
    model.fc = nn.Identity() #remove class head, get the 2048 dim feature embedding (before class)!
    #freeze params
    for param in model.parameters():
        param.requires_grad = False
        
    return model.to(device).eval()

#according to inception_v3 documentation (pretrained on imagenet data (=imagenet mean&std dev norm)
inv3_transform = T.Compose([
    T.Resize((299,299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])

# === processing functions ====================================
''' this was inefficient (tile to 4 patches)
#from h5 get a 598x598x3 numpy tile. inception expects 299x299x3 input
def split_tile_input_backbone(tile):
    h, w = tile.shape[:2]
    assert h == 598 and w == 598, 'Expect tile size 598x598'

    tile_patches = [] #store 4 x 299x299x3 patches per tile 
    for i in [0, 299]:
        for j in [0, 299]:
            patch = tile[i:i+299 , j:j+299, :]
            tile_patches.append(patch)
    return tile_patches
'''

#load h5 (N tiles of 1 WSI) -> return 1 aggregated feature vector
def extract_feat_from_h5(h5_path, feat_extract_model, device, num_tiles):
    file_id = os.path.splitext(os.path.basename(h5_path))[0]
    try:
        #tile-wise 'lazy' access to the tile stack of a WSI:
        with h5py.File(h5_path, 'r') as f:
            if file_id not in f or 'tiles' not in f[file_id]:
                print(f'No Tiles: Skip {h5_path}')
                return None , file_id
            
            tile_stack = f[file_id]['tiles'] # N,598,598,3 = N tiles representing 1WSI
            num_tiles_in_stack = tile_stack.shape[0]

            if num_tiles_in_stack == 0:
                print(f'No Tiles (Empty Tile Stack): Skip {h5_path}')
                return None, file_id

            #random select 8 tiles
            indices = np.random.choice(num_tiles_in_stack, size=min(num_tiles, num_tiles_in_stack), replace=False)
            tiles = [tile_stack[i] for i in indices]
            
            tile_batch = []
            for tile in tiles:                              
                tile_img = Image.fromarray(tile.astype(np.uint8))
                tile_tensor = inv3_transform(tile_img) #shape: 3,299,299 (.unsqueeze(0) -> #1,3,299,299 use unsqueeze only if single patch processed (adds batch dimension)!)
                tile_batch.append(tile_tensor)
                
            tile_batch = torch.stack(tile_batch).to(device) #8,3,299,299 
                
            with torch.no_grad():
                tile_features = feat_extract_model(tile_batch) #8,2048
            
            if tile_features is None or tile_features.shape[0]==0:
                print(f'No features extracted for {h5_path}. Skipped')
                return None , file_id
            
            tile_features_mean = tile_features.mean(dim=0) # aggregate to 1 feature vector for 1 WSI -> shape 2048, 
            return tile_features_mean.cpu().numpy() , file_id
        
    except Exception as e:
        print(f'Error {file_id}: {e}')
        return None , file_id