import cv2
import numpy as np
import os
import pandas as pd
import argparse
import toml

from utils.load_tif import *
from utils.load_mask import *
from utils.tile_img import *
from utils.percentile_stretch import *



def parse_args():
    parser = argparse.ArgumentParser(description="Tile satellite images")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to TOML configuration file"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = args.config_file
    config = toml.load(config_path)

    tile_size = config["tile_size"]
    output_directory = config["output_directory"]
    raster_directories = config["raster_directories"]
    

    # Loop through all raster images
    for directory in raster_directories:
        contents = os.listdir(directory)
        tif_name = directory.split("\\")[-1]
        tif_path = None
        shp_path = None
        print(f"Processing: {tif_name}")

        # Populate .tif and .shp filepaths
        for filename in contents:
            if filename.endswith(".tif"):
                tif_path = os.path.join(directory, filename)
            elif filename.endswith(".shp"):
                shp_path = os.path.join(directory, filename)
        
        if tif_path is None or shp_path is None:
            print(f"Warning: missing .tif or .shp in {directory}")
            continue
        
        # Load and Process .shp
        mask = load_mask(tif_path, shp_path)
        mask = (mask > 0).astype(np.uint8) * 255

        # Load and Process .tif
        [img, profile] = load_tif(tif_path)                         # Load .tif
        img = np.transpose(img, (1, 2, 0))                          # Reshape from (C,H,W) -> (H,W,C)
        max_vals = np.max(img, axis=(0, 1))
        print(f"\tChannel max vals: {max_vals}")
        means = np.mean(img, axis=(0, 1))
        print("\tChannel means:", means)

        assert img.shape[:2] == mask.shape

        RGB_IDX = [0, 1, 2]   # or [2,1,0], etc
        NIR_IDX = 3
        img = img[:, :, RGB_IDX].astype(np.float32)
        img = np.where(img == 0, np.nan, img)

        # Cut off mask at image edges
        mask_cutoff = np.all(img == 0, axis=2)
        mask[mask_cutoff] = 0
        cv2.imwrite(f"test_dmask.png", mask)

        # Perform percentile stretch
        for i in range(3):
            print(f"\tStretching channel {i}")
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
        img_write_dir = os.path.join(output_directory, tif_name, "img")
        mask_write_dir = os.path.join(output_directory, tif_name, "mask")
        os.makedirs(img_write_dir, exist_ok=True)
        os.makedirs(mask_write_dir, exist_ok=True)
        
        filename_array = []
        img_path_array = []
        mask_path_array = []

        halo_present_array = []
        num_objects_array = []
        coords_array = []

        # Loop through tiles
        for i, img_tile in enumerate(img_tiles):
            temp_tile_name = f"{tif_name}_{i}.png"
            temp_img_path = os.path.join(img_write_dir, temp_tile_name)
            temp_mask_path = os.path.join(mask_write_dir, temp_tile_name)

            # Show only valid tiles
            if (np.mean(img_tile > 10) > 0.001):
                filename_array.append(temp_tile_name)
                img_path_array.append(temp_img_path)
                mask_path_array.append(temp_mask_path)
                coords_array.append(coords[i])
                #print(temp_img_path)
                #print(temp_mask_path)
                mask_tile = mask_tiles[i]

                # If object present in mask
                if np.max(mask_tile) == 255:
                    halo_present_array.append(True)
    
                    num_labels, labels = cv2.connectedComponents(mask_tile)         # Count connected components (subtract 1 to exclude background label)
                    num_objects = num_labels - 1
                    num_objects_array.append(num_objects)

                else:
                    halo_present_array.append(False)
                    num_objects_array.append(0)

                print(f"\tWriting to: {temp_img_path}")
                cv2.imwrite(temp_img_path, img_tile)
                cv2.imwrite(temp_mask_path, mask_tile)

        df = pd.DataFrame({
            "filename": filename_array,
            "img_path": img_path_array,
            "mask_path": mask_path_array,
            "coord": coords_array,
            "tile_size": tile_size,
            "halo_present": halo_present_array,
            "num_objects": num_objects_array,
        })

        print(f"Writing .csv")
        temp_dataset_path = os.path.join(output_directory, f"{tif_name}_dataset.csv")
        df.to_csv(temp_dataset_path, index=False)
        val_counts = df["halo_present"].value_counts()
        print(val_counts)

if __name__ == "__main__":
    main()