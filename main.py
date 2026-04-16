import cv2
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
import pandas as pd
import toml

from utils.load_tif import *
from utils.load_mask import *
from utils.tile_img import *
from utils.percentile_stretch import *

data_dir = "./data/"

for file in os.listdir(data_dir):
    if file.endswith(".tif"):
        temp_img_path = os.path.join(data_dir, file)        # Create temp path
        temp_img_path = os.path.join(data_dir, "BLZ01.tif")
        temp_shp_path = os.path.join(data_dir, "LabeledHalos_BLZ01.shp")  
        
        # Load and Process .shp
        mask = load_mask(temp_img_path, temp_shp_path)
        mask = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(f"test_dmask.png", mask)
        #print(np.max(mask))
        #print(mask.dtype)

        # Load and Process .tif
        [img, profile] = load_tif(temp_img_path)            # Load .tif
        img = np.transpose(img, (1, 2, 0))              # Reshape from (C,H,W) -> (H,W,C)
        max_vals = np.max(img, axis=(0, 1))
        print(f"\tChannel max vals: {max_vals}")
        means = np.mean(img, axis=(0, 1))
        print("\tChannel means:", means)

        assert img.shape[:2] == mask.shape

        RGB_IDX = [0, 1, 2]   # or [2,1,0], etc
        NIR_IDX = 3
        img = img[:, :, RGB_IDX].astype(np.float32)
        img = np.where(img == 0, np.nan, img)

        # Perform percentile stretch
        for i in range(3):
            channel = img[:, :, i]
            stretched_channel = percentile_stretch(channel)
            img[:, :, i] = stretched_channel

        # Convert invalid values to int
        img = np.nan_to_num(
            img, 
            nan=0.0, 
            posinf=0.0, 
            neginf=0.0
        )
        
        # Normalize to 255 and performing tiling
        tile_size = 512
        img = (img * 255).astype(np.uint8)
        [img_tiles, coords, dims] = tile_img(
            img = img,
            tile_size = tile_size
        )
        [mask_tiles, _, _] = tile_img(
            img = mask,
            tile_size = tile_size
        )


        # Create output directory
        out_dir = "./output"
        img_dir = os.path.join(out_dir, "img")
        mask_dir = os.path.join(out_dir, "mask")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        filename_array = []
        img_path_array = []
        mask_path_array = []

        halo_present_array = []
        coords_array = []

        # Loop through tiles
        for i, img_tile in enumerate(img_tiles):
            temp_file_name = f"{file.split('.')[0]}_{i}.png"
            temp_img_path = os.path.join(img_dir, temp_file_name)
            temp_mask_path = os.path.join(mask_dir, temp_file_name)

            # Show only valid tiles
            if (np.mean(img_tile > 10) > 0.001):
                filename_array.append(temp_file_name)
                img_path_array.append(temp_img_path)
                mask_path_array.append(temp_mask_path)
                coords_array.append(coords[i])
                #print(temp_img_path)
                #print(temp_mask_path)
                mask_tile = mask_tiles[i]

                # If object present in mask
                if np.max(mask_tile) == 255:
                    halo_present_array.append(True)
                else:
                    halo_present_array.append(False)

                cv2.imwrite(temp_img_path, img_tile)
                cv2.imwrite(temp_mask_path, mask_tile)
                #cv2.imshow(temp_mask_path, mask_tile)
                #cv2.imshow(temp_img_path, img_tile)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
        df = pd.DataFrame({
            "filename": filename_array,
            "img_path": img_path_array,
            "mask_path": mask_path_array,
            "halo_present": halo_present_array,
            "coord": coords_array
        })

        df.to_csv("dataset.csv", index=False)
        #cv2.imwrite(f"test.png", img)
    

        
        input('pause')