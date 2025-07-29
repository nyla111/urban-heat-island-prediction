# Configuration constants
BOUNDS = (105.932202, 20.983959, 105.962790, 21.004831)  # (min_lon, min_lat, max_lon, max_lat)
TIME_WINDOW = "2020-01-01/2025-12-01"
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTIONS = ["landsat-c2-l2"]
RESOLUTION = 30  # meters
SCALE = RESOLUTION / 111320.0  # degrees per pixel for crs=4326

# Band configurations
BANDS = {
    "main_bands": ["coastal", "red", "green", "blue", "nir08", "swir16"],
    "thermal_band": ["lwir11"]
}

# Scaling factors
SCALE_FACTORS = {
    "rgb_nir": {
        "scale": 0.0000275,
        "offset": -0.2
    },
    "thermal": {
        "scale": 0.00341802,
        "offset": 149.0,
        "kelvin_celsius": 273.15
    }
}

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 123