import numpy as np
import pandas as pd

import json
import requests
import os
from pathlib import Path

import openslide
import cv2
import h5py
#import hdf5plugin #for blosc compression

from concurrent.futures import ProcessPoolExecutor , as_completed
import psutil #needed in check ram function
import time #needed in check ram function
import gc #python garbage collector

import logging
import multiprocessing
import sys

# Paths / settings ================================
project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data'
file_id_path = data_dir / "tcga_brca_file_ids.csv"
cases_files = pd.read_csv(file_id_path)
file_ids = cases_files["file_id"].tolist()

svs_dir = data_dir / "temp_svs"
metadata_dir = data_dir / "metadata"
wsi_thumbnail_dir = data_dir / "thumbnails" / "wsi"
mask_thumbnail_dir = data_dir / "thumbnails" / "mask"
h5_dir = data_dir / "tiles"
for path in [svs_dir, metadata_dir, wsi_thumbnail_dir, mask_thumbnail_dir, h5_dir]:
    path.mkdir(parents=True, exist_ok=True)
'''
svs_dir = "./temp_svs"
metadata_dir = "./metadata"
wsi_thumbnail_dir = "./thumbnails/wsi"
mask_thumbnail_dir = "./thumbnails/mask"
h5_dir = "./tiles"

os.makedirs(svs_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(wsi_thumbnail_dir, exist_ok=True)
os.makedirs(mask_thumbnail_dir, exist_ok=True)
os.makedirs(h5_dir, exist_ok=True)
'''
mag_tolerance = 0.01

data_ep = "https://api.gdc.cancer.gov/data/"
#functions ===================================================

def setup_logger(name="wsi_logger", log_file="wsi_preprocess.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # StreamHandler (stdout)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter('[%(asctime)s][%(processName)s][%(levelname)s] %(message)s'))

        # FileHandler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('[%(asctime)s][%(processName)s][%(levelname)s] %(message)s'))

        logger.addHandler(sh)
        logger.addHandler(fh)

    return logger 

def check_memory(min_ram_gb=64, max_wait_sec=600):
    start_time = time.time()
    while psutil.virtual_memory().available < min_ram_gb * 1e9:
        if time.time() - start_time > max_wait_sec:
            raise TimeoutError(f"Waited over {max_wait_sec}s for free RAM.")
        logger = logging.getLogger("wsi_logger")
        logger.info("Low RAM. Waiting before processing...")
        time.sleep(10)  # Wait and recheck  

def download_svs (file_id, svs_path, metadata_path , h5_path , wsi_thumbnail_path , mask_thumbnail_path):
    
    if metadata_path.exists() and h5_path.exists() and wsi_thumbnail_path.exists() and mask_thumbnail_path.exists():
        logger = logging.getLogger("wsi_logger")
        logger.info(f"skipping {file_id}: Metadata, Tiles, Thumbnails already exist")
        return None
    
    if svs_path.exists():
        logger = logging.getLogger("wsi_logger")
        logger.info(f"{svs_path} file already exists: skip download")
        return svs_path
    
    try:
        start_downl_time = time.time()
        response = requests.get(data_ep + file_id, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(svs_path, "wb") as file: #w for writing, b binary for images
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger = logging.getLogger("wsi_logger")
        logger.info(f"download complete: {svs_path} , download time {time.time() - start_downl_time:.2f} sec.")
        return svs_path
    
    except requests.exceptions.RequestException as e:
        logger = logging.getLogger("wsi_logger")
        logger.info(f"download failed: {file_id} , error: {e}")
        if svs_path.exists():
            svs_path.unlink()
        return None
    
def extract_wsi(svs_path, file_id, metadata_path, tol):

    with openslide.OpenSlide(svs_path) as slide:

        metadata = {key : slide.properties[key] for key in slide.properties.keys()}
        metadata["level_count"] = slide.level_count
        metadata["level_dimensions"] = slide.level_dimensions
        metadata["level_downsamples"] = slide.level_downsamples

        target_mag = 20
        
        base_mag = int(metadata.get("openslide.objective-power", 40))
        scale_factor = base_mag / target_mag
        best_level = slide.get_best_level_for_downsample(scale_factor)
        if best_level is None:
            raise ValueError(f"no magnification >= {target_mag}x in {svs_path}")

        best_downsample = slide.level_downsamples[best_level]
        best_mag = base_mag / best_downsample
        best_dims = slide.level_dimensions[best_level]

        requires_downsampling = False

        if best_mag > target_mag and abs(best_mag - target_mag) > tol:
            requires_downsampling = True  

        wsi = slide.read_region((0, 0), best_level, best_dims).convert("RGB") # = Pillow image object! 
        
        wsi = np.array(wsi) #(H,W,3) uint8 / RGB , not needed  if downsampling via pillow (openslide extracts pillow object already)
        
        downsample_factor = None

        if requires_downsampling:
            downsample_factor = target_mag / best_mag
            new_size = (int(wsi.shape[1] * downsample_factor) , int(wsi.shape[0] * downsample_factor)) #use width=shape[1] abd height=shape[0] if wsi is numpy array

            #wsi = np.array(wsi)
            wsi = cv2.resize(wsi, new_size , interpolation=cv2.INTER_LINEAR) #returns np array H,W,3 uint8
            wsi = cv2.cvtColor(wsi, cv2.COLOR_RGB2BGR) #activate this if using cv2.imwrite() below to save png
            #wsi = Image.fromarray(wsi) activate if using pillow to save png

            final_mag = target_mag
        else:
            wsi = cv2.cvtColor(wsi, cv2.COLOR_RGB2BGR) #activate this if using cv2.imwrite() below to save png
            final_mag = best_mag

        #wsi = np.array(wsi) #(H,W,3) in RGB (uint8), needed fo pillow object only
                
        metadata["Base Magnification"] = base_mag
        metadata["Extracted Level"] = best_level
        metadata["Extracted Downsample"] = best_downsample
        metadata["Extracted Magnification"] = best_mag
        metadata["Extracted Dimensions"] = best_dims
        metadata["Downsampling Applied"] = requires_downsampling
    
        metadata["Downsampling Factor"] = downsample_factor
        metadata["Final Magnification"] = final_mag
        metadata["Final Dimensions"] = [wsi.shape[1], wsi.shape[0]] #[wsi.width , wsi.height] #if numpy change to height, width
    
        with open(metadata_path, "w") as file:
            json.dump(metadata, file, indent=4)
        logger = logging.getLogger("wsi_logger")
        logger.info(f"saved metadata {metadata_path}")

        #slide.close()
    return wsi

def make_tiles(img, wsi_thumbnail_path, mask_thumbnail_path, size=(598 , 598)):
    h, w = img.shape[:2]

    #via Otsu thresholding create a binary mask ( 0 = black = tissue , 255 = white = bg) to segment background vs tissue 
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU ) #outputs: threshold limit , mask

    #make thumbnails
    max_dim = 1024
    scale = max_dim / max(h, w)
    thumb_dim = (int(w*scale), int(h*scale))
    thumb_wsi = cv2.resize(img, thumb_dim, interpolation=cv2.INTER_AREA)
    thumb_mask = cv2.resize(mask, thumb_dim, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(wsi_thumbnail_path, thumb_wsi, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(mask_thumbnail_path, thumb_mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    tiles = []
    positions = []

    for y in range(0, h, size[1]):
        for x in range(0, w, size[0]):
            tile = img[y:y+size[1], x:x+size[0]]
            mask_tile = mask[y:y+size[1], x:x+size[0]]

            #only keep 598x598 tiles (discards WSI right edge & bottom edge tiles, that are likely to are backgopund anyways)
            if tile.shape[:2] == (size[1] , size[0]):
                #select tissue tiles (>80% tissue) #### test 60%
                if np.sum(mask_tile == 0) / mask_tile.size > 0.8:
                    tiles.append(tile)
                    positions.append((x,y))
        
    return tiles, positions

def save_h5(h5_path, file_id, tiles, positions):
    try:
        tiles = np.stack(tiles) # ( N , H , W , C)
        positions = np.array(positions , dtype = np.int32) # (N,2)

        with h5py.File(h5_path, 'w') as h5f:
            grp = h5f.create_group(file_id)
        
            grp.create_dataset( "tiles" , data= tiles, compression = "gzip" , chunks=(1, *tiles.shape[1:]))
        
            grp.create_dataset("positions", data=positions, dtype=np.int32)
            grp.attrs["tile_count"] = len(tiles)
    
    except Exception as e:
        logger = logging.getLogger("wsi_logger")
        logger.error(f"No Tiles saved into H5: {file_id}: {e}", exc_info=True)



########## The function below will run the full preprocessing (using functions defined above) ######################################
def wsi_tiling(file_id):
    svs_path = svs_dir / f"{file_id}.svs"
    metadata_path = metadata_dir / f"{file_id}.json"
    tol = mag_tolerance
    h5_path = h5_dir / f"{file_id}.h5"
    wsi_thumbnail_path = wsi_thumbnail_dir / f"{file_id}_wsi.png"
    mask_thumbnail_path = mask_thumbnail_dir / f"{file_id}_mask.png"

    svs_path = download_svs (file_id, svs_path, metadata_path , h5_path , wsi_thumbnail_path , mask_thumbnail_path)
    check_memory(min_ram_gb=10)

    if svs_path:
        start_processing_time = time.time()

        try:
            wsi = extract_wsi(svs_path, file_id, metadata_path , tol)

            tiles , positions = make_tiles(img=wsi, wsi_thumbnail_path = wsi_thumbnail_path, mask_thumbnail_path = mask_thumbnail_path) 
            
            save_h5(h5_path= h5_path,
                    file_id= file_id,
                    tiles= tiles,
                    positions = positions)
          
            logger = logging.getLogger("wsi_logger")
            logger.info(f"{file_id}: {len(tiles)} tiles saved to {h5_path}")

            del wsi
            del tiles
            del positions
            gc.collect()
        
        except Exception as e:
            logger = logging.getLogger("wsi_logger")
            logger.error(f"Error processing {file_id}: {e}", exc_info=True)
            return
        
        finally:
            try:
                if metadata_path.exists() and h5_path.exists() and wsi_thumbnail_path.exists() and mask_thumbnail_path.exists():
                    if svs_path.exists():
                        svs_path.unlink()
                        logger = logging.getLogger("wsi_logger")
                        logger.info(f"deleted svs {svs_path}")
            except Exception as e:
                logger = logging.getLogger("wsi_logger")
                logger.warning(f"Failed to delete svs: {svs_path}: {e}")
            
        logger = logging.getLogger("wsi_logger")
        logger.info(f"processing complete: {file_id} , process. time {time.time() - start_processing_time:.2f} sec.")
        gc.collect()

if __name__ == "__main__":
    
    multiprocessing.set_start_method("fork")

    logger = setup_logger()

    os.environ["OMP_NUM_THREADS"] = "16"
    cv2.setNumThreads(16)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=4 , initializer= setup_logger) as executor:
        executor.map(wsi_tiling, file_ids)

    logger.info(f"total time {time.time() - start_time:.2f} sec.")