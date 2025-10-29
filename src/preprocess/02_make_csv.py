#import os
import json
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data'
file_id_path = data_dir / "tcga_brca_file_ids.csv"
df = pd.read_csv(file_id_path)
file_ids = df["file_id"].tolist()

tiles_dir = data_dir / "tiles"
metadata_dir = data_dir / "metadata"


def get_metadata(metadata_path):

    try:
        metadata = json.load(open(metadata_path, "r"))

    except (FileNotFoundError, json.JSONDecodeError):
        print(f"could not load metadata {metadata_path}. skipping.")
        return None

 
    extr_level = metadata.get("Extracted Level") 
    
    #the following fields are partially retrieved directly from the metadata of the WSI SVS, and from the preprocessing steps
    extracted_metadata = {
        "Vendor": metadata.get("openslide.vendor", "Unknown"),
        "Original Magnification": int(metadata.get("openslide.objective-power", 40)),
        "MPP X": metadata.get("openslide.mpp-x", "Unknown"),
        "MPP Y": metadata.get("openslide.mpp-y", "Unknown"),
        "Extracted Level": extr_level,
        "Extracted Magnification": metadata.get("Extracted Magnification"),
        "Extracted Dimensions": metadata.get("Extracted Dimensions"),
        "Extracted Level Downsample": metadata.get("Extracted Downsample"),
        "Tile Width": metadata.get(f"openslide.level[{extr_level}].tile-width", "Unknown") if extr_level is not None else "Unknown",
        "Tile Height": metadata.get(f"openslide.level[{extr_level}].tile-height", "Unknown") if extr_level is not None else "Unknown",
        "Downsampling Applied": metadata.get("Downsampling Applied"),
        "Applied Factor": metadata.get("Downsampling Factor", "Unknown"),
        "Final Magnification": metadata.get("Final Magnification" , "Unknown"),
        "Final Dimensions": metadata.get("Final Dimensions" , "Unknown")
    }
       
    #print(json.dumps(extracted_metadata, indent=4))
    return extracted_metadata

def update_csv_metadata(df, tiles_path, metadata_path, file_id):
    
    if not tiles_path.exists() or not metadata_path.exists() :
        print(f"skip {file_id}: missing WSI (Tiles) or metadata file")
        return df
    
    extr_metadata = get_metadata(metadata_path)

    df.loc[df["file_id"] == file_id , "tiles_path"] = tiles_path
    df.loc[df["file_id"] == file_id , "metadata_path"] = metadata_path

    for key , value in extr_metadata.items():
        df.loc[df["file_id"] == file_id , key] = str(value)
    
    return df


for file in file_ids:
    metadata_path = metadata_dir / f"{file}.json"
    tiles_path = tiles_dir / f"{file}.h5"
    df_meta = update_csv_metadata(df=df,
                             tiles_path=tiles_path,
                             metadata_path=metadata_path,
                             file_id=file)

#print(df_meta.head())
out_path = data_dir / "tcga_brca_file_ids_metadata.csv"
df_meta.to_csv(out_path, index=False)