import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from ..config import PLOT_COLORS


def plot_rgb(data, scene_idx=1, vmin=0.0, vmax=0.25):
    """Plot RGB image for a specific scene"""
    fig, ax = plt.subplots(figsize=(9,10))
    data.isel(time=scene_idx)[["red", "green", "blue"]].to_array().plot.imshow(
        robust=True, ax=ax, vmin=vmin, vmax=vmax)
    ax.set_title("RGB Real Color")
    ax.axis('off')
    plt.show()


def plot_median_composite(median_data, vmin=0.0, vmax=0.25):
    """Plot median composite"""
    fig, ax = plt.subplots(figsize=(6,6))
    median_data[['red', 'green', 'blue']].to_array().plot.imshow(
        robust=True, ax=ax, vmin=vmin, vmax=vmax)
    ax.set_title("RGB Median Composite")
    ax.axis('off')
    plt.show()


def plot_ndvi(ndvi_data, vmin=0.0, vmax=1.0):
    """Plot NDVI data"""
    fig, ax = plt.subplots(figsize=(11,10))
    ndvi_data.plot.imshow(vmin=vmin, vmax=vmax, cmap="RdYlGn")
    plt.title("Vegetation Index = NDVI")
    plt.axis('off')
    plt.show()
    

def plot_lst(lst_data, vmin=20.0, vmax=45.0):
    """Plot land surface temperature"""
    fig, ax = plt.subplots(figsize=(11,10))
    lst_data.plot.imshow(vmin=vmin, vmax=vmax, cmap="jet")
    plt.title("Land Surface Temperature (LST)")
    plt.axis('off')
    plt.show()