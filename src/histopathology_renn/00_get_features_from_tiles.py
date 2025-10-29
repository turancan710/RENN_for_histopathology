#import os
import numpy as np
import time
import torch
from histo_feat_extr_backbone import extract_feat_from_h5 , get_inception_v3_feature_extractor
from histo_utils import global_seed
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {device}")

init_seed = 62
global_seed(init_seed)

num_tiles = 32

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data'
h5_dir = data_dir / 'tiles'

out_dir = data_dir / 'incv3_frozen_features'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'features.npz'

model = get_inception_v3_feature_extractor(device)

#h5_files = [f for f in h5_dir.iterdir() if f.suffix == '.h5'] #old, file names only
h5_files = list(h5_dir.glob("*.h5")) #full paths to files

print(f'extracting {num_tiles} from each WSI (for {len(h5_files)} WSIs (h5 files))')
print(f'Backbone: Frozen INCEPTION_V3 (Imagenet pretrained) from torchvision')
features = {}
start_time = time.time()

for i , h5_path in enumerate(h5_files):
    #h5_path = h5_dir / h5_file #old: merge dir + file names
    feat , file_id = extract_feat_from_h5(h5_path , model, device, num_tiles)
    if feat is not None:
        features[file_id] = feat #store a 2048 dim feat vector per file_id
    #update e.g. every 10 files
    if (i+1) % 10 == 0:
        print(f'processed {i+1}/{len(h5_files)}')

end_time = time.time()
print(f'Feature extraction finished in {end_time - start_time} sec.')
print(f'Succesfully extracted features for: {len(features)} WSI')

np.savez_compressed(out_path, **features) #for dict of numpy arrays i.e. file_id : feature_vector_numpy
print(f'saved features to {out_path}')