from pystac_client import Client
from odc.stac import stac_load
import planetary_computer as pc
import xarray as xr
import numpy as np
from ..config import BOUNDS, TIME_WINDOW, STAC_API_URL, COLLECTIONS, SCALE


def load_stac_data():
    """Load and preprocess STAC data"""
    stac = Client.open(STAC_API_URL)
    
    search = stac.search(
        bbox=BOUNDS,
        datetime=TIME_WINDOW,
        collections=COLLECTIONS,
        query={"eo:cloud_cover": {"lt": 50}, "platform": {"in": ["landsat-8"]}},
    )
    
    items = list(search.get_items())
    print(f'Number of scenes: {len(items)}')
    
    # Load main bands
    data1 = stac_load(
        items,
        bands=BANDS["main_bands"],
        crs="EPSG:4326",
        resolution=SCALE,
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=pc.sign,
        bbox=BOUNDS
    )
    
    # Load thermal band
    data2 = stac_load(
        items,
        bands=BANDS["thermal_band"],
        crs="EPSG:4326",
        resolution=SCALE,
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=pc.sign,
        bbox=BOUNDS
    )
    
    return data1, data2, items


def scale_data(data1, data2):
    """Apply scaling to datasets"""
    # Scale RGB/NIR bands
    data1 = data1.astype(float) * SCALE_FACTORS["rgb_nir"]["scale"] + SCALE_FACTORS["rgb_nir"]["offset"]
    
    # Scale thermal band
    data2 = (data2.astype(float) * SCALE_FACTORS["thermal"]["scale"] + 
             SCALE_FACTORS["thermal"]["offset"] - SCALE_FACTORS["thermal"]["kelvin_celsius"])
    
    return data1, data2


def filter_scenes(data1, data2, indexes_to_remove):
    """Remove low-quality scenes"""
    data1 = data1.isel(time=~data1.time.isin(data1.time[indexes_to_remove]))
    data2 = data2.isel(time=~data2.time.isin(data2.time[indexes_to_remove]))
    return data1, data2