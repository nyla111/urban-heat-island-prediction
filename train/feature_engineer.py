import xarray as xr
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from ..config import SCALE_FACTORS


def calculate_ndvi(median_data):
    """Calculate NDVI from median composite"""
    return (median_data.nir08 - median_data.red) / (median_data.nir08 + median_data.red)


def calculate_uhi(ndvi_data, lst_data):
    """
    Calculate UHI index per urban pixel using xarray DataArrays for NDVI and LST.
    Returns normalized UHI map where UHI is computed only for urban pixels.
    """
    global_mean_lst = lst_data.mean(skipna=True)
    normalized_uhi = lst_data / global_mean_lst
    
    normalized_uhi.name = "Normalized_UHI"
    normalized_uhi.attrs["units"] = "relative index"
    normalized_uhi.attrs["description"] = (
        "Normalized Urban Heat Island intensity (UHI / mean UHI in urban areas). "
        "<1 = cooler-than-average, >1 = hotter-than-average."
    )
    return normalized_uhi
    

def prepare_dataframe(median1, median2, ndvi_data, uhi_map):
    """Prepare DataFrame for machine learning"""
    # Convert xarray DataArrays to NumPy arrays
    coastal_values = median1.coastal.values.flatten()
    red_values = median1.red.values.flatten()
    green_values = median1.green.values.flatten()
    blue_values = median1.blue.values.flatten()
    nir08_values = median1.nir08.values.flatten()
    swir16_values = median1.swir16.values.flatten()
    lwir11_values = median2.lwir11.values.flatten()
    ndvi_values = ndvi_data.values.flatten()
    uhi_values = uhi_map.values.flatten()

    return pd.DataFrame({
        'coastal': coastal_values,
        'red': red_values,
        'green': green_values,
        'blue': blue_values,
        'nir08': nir08_values,
        'swir16': swir16_values,
        'lwir11': lwir11_values,
        'NDVI': ndvi_values,
        'UHI': uhi_values
    })