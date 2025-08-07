# Urban Heat Island Prediction in Hanoi Using Satellite Imagery and Machine Learning

This project presents a geospatial and machine learning pipeline to predict Urban Heat Island (UHI) intensity over Hanoi, Vietnam. Using multi-temporal Landsat-8 imagery and derived vegetation indices, we build predictive models to estimate UHI severity and apply SHAP analysis for interpretability.

---

## Project Objective

To analyze and predict the intensity of urban heat islands in the Ocean Park I (OCPI) region of Hanoi using satellite-derived environmental features and state-of-the-art machine learning techniques, including temporal modeling and explainability.

---

## Data Overview

- **Satellite Sources**:  
  - Landsat-8 (Collection 2, Level-2)
  - [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)

- **Time Range**:  2020 - 2025

- **Spatial Area**:  Ocean Park I (Hanoi):  
    Lower-left: `(20.983959, 105.932202)`  
    Upper-right: `(21.004831, 105.962790)`

- **Cloud Filter**:  Less than 50% cloud coverage

---

## Methodology & Workflow

1. **Image Retrieval**  
   Query Landsat-8 Level-2 data via STAC API, applying date range and cloud filter.

2. **Preprocessing**  
   Load GeoTIFF imagery using `odc.stac` and apply radiometric and thermal scaling. Generate median composites to remove transient noise (clouds, shadows).

3. **Index Computation**  
   - NDVI (vegetation)
   - LST (Land Surface Temperature)
   - Normalized UHI intensity

4. **Feature Engineering**  
   - Input features: B01 (Coastal), B02 - B06 (Blue - SWIR), NDVI  
   - Target: Normalized UHI (from LWIR11 band)

5. **Modeling**  
   - Baseline: Random Forest  
   - Enhanced: XGBoost  
   - Deep Learning: LSTM for temporal modeling  
   - Evaluation: R², RMSE, MAE

6. **Explainability (SHAP Analysis)**  
   Quantify the influence of each input feature on UHI predictions. Identify key spectral drivers and urban-environmental correlations.

---

## Key Findings

- **Best Model**: LSTM (R² ≈ 0.87)  
- **SHAP Insights**: NDVI and SWIR bands are dominant indicators of UHI severity   
- **Preprocessing**: Cloud filtering and proper scaling were essential to performance

---

## Visual Results

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Random Forest | 0.6607 | 0.0246 | 0.0328 |
| XGBoost       | 0.6474 | 0.0259 | 0.0334 |
| LSTM          | 0.8868 | 0.0134 | 0.0185 |

> See `/results/` folder for visual plots and SHAP diagrams.

---

## Tools & Libraries

| Category        | Tools / Libraries                                  |
|----------------|-----------------------------------------------------|
| Data Access     | Microsoft Planetary Computer (STAC API)            |
| Query & Load    | `pystac_client`, `odc.stac`, `xarray`              |
| Processing      | `numpy`, `pandas`, `dask`, `matplotlib`, `seaborn`|
| Modeling        | `scikit-learn`, `xgboost`, `tensorflow` (LSTM)     |
| Explainability  | `shap`                                             |
| Scaling         | `RobustScaler` (scikit-learn)                      |

---

## Project Structure
```
UHI/
├── data/
│   └── load_data.py              # Functions to load and preprocess raw satellite and ground truth data
│
├── results/
│   ├── shap_analysis.py         # SHAP explainability for model interpretability
│   └── visualization.py         # Visualization utilities for model outputs and indices
│
├── train/
│   ├── feature_engineer.py      # Feature extraction and transformation logic (e.g., NDVI, scaling)
│   ├── ltsm.py                  # LSTM deep learning model architecture and training pipeline
│   ├── models.py                # Traditional ML models (Random Forest, XGBoost) and utilities
│
├── config.py                    # Configuration variables (e.g., paths, hyperparameters)
├── dependencies.py              # Dependency checking and environment setup
├── main.py                      # Entry point to run full training and evaluation pipeline
│
└── requirements.txt             # All required Python packages with pinned versions
│
└── README.md
```
