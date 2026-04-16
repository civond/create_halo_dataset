import rasterio
import geopandas as gpd
from rasterio.features import rasterize

def load_mask(img_pth, shp_pth, class_col="class"):
    """
    Rasterize shapefile into mask aligned with GeoTIFF image.
    
    Args:
        img_pth (str): path to reference GeoTIFF
        shp_pth (str): path to shapefile (.shp)
        class_col (str): column in .dbf containing class labels
    
    Returns:
        mask (np.ndarray): H x W label mask
    """

    # --- Load reference image (defines grid) ---
    with rasterio.open(img_pth) as src:
        img_shape = src.shape          # (H, W)
        transform = src.transform
        img_crs = src.crs

    # --- Load shapefile ---
    gdf = gpd.read_file(shp_pth)

    # --- Ensure CRS alignment ---
    if gdf.crs != img_crs:
        gdf = gdf.to_crs(img_crs)

    # --- Build (geometry, value) pairs ---
    if class_col in gdf.columns:
        shapes = list(zip(gdf.geometry, gdf[class_col]))
    else:
        # fallback: binary mask (all 1s)
        shapes = [(geom, 1) for geom in gdf.geometry]

    # --- Rasterize ---
    mask = rasterize(
        shapes=shapes,
        out_shape=img_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    return mask