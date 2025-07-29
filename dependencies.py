# Installations
!pip install rioxarray stackstac pystac_client planetary_computer odc-stac rasterio shap xgboost torch

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Xarray and geospatial
import xarray as xr
import rioxarray as rxr

# Planetary Computer & STAC
import planetary_computer as pc
import pystac_client
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
from odc.stac import stac_load
import stackstac
import rasterio

# Geospatial
import geopandas as gpd

# Machine learning
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# SHAP and deep learning
import shap
import torch
import torch.nn as nn
import torch.optim as optim