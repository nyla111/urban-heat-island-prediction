from data.load_data import load_stac_data, scale_data, filter_scenes
from train.feature_engineer import calculate_ndvi, calculate_uhi, prepare_dataframe
from train.models import train_random_forest, train_xgboost, evaluate_model
from train.ltsm import LSTMModel, prepare_lstm_data, train_lstm
from results.visualization import plot_rgb, plot_median_composite, plot_ndvi, plot_lst
from results.shap_analysis import perform_shap_analysis, compare_shap_explanations
from config import *
import numpy as np
import pandas as pd


def main():
    # 1. Data Loading
    data1, data2, items = load_stac_data()
    data1, data2 = scale_data(data1, data2)
    
    # 2. Data Filtering
    indexes_to_remove = [5,6,7,12,14,15,17,20,21,22,27,31,33,34,36,37,38,39,40,41,42,43,45,46,47,48,51,52,54,58,60,61,62,63]
    data1, data2 = filter_scenes(data1, data2, indexes_to_remove)
    
    # 3. Visualization
    plot_rgb(data1, scene_idx=1)
    
    # 4. Feature Engineering
    median1 = data1.median(dim="time").compute()
    median2 = data2.where(data2 >= 0).median(dim="time").compute()
    plot_median_composite(median1)
    
    ndvi_data = calculate_ndvi(median1)
    plot_ndvi(ndvi_data)
    
    plot_lst(median2.lwir11)
    
    uhi_map = calculate_uhi(ndvi_data, median2.lwir11)
    uhi_data = prepare_dataframe(median1, median2, ndvi_data, uhi_map)
    
    # 5. Model Training
    X_train, X_test, y_train, y_test = prepare_train_test(uhi_data)
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_train, y_train, X_test, y_test)
    
    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_train, y_train, X_test, y_test)
    
    # 6. LSTM Model
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_scaler, y_scaler = prepare_lstm_data(uhi_data)
    
    lstm_model = LSTMModel(
        input_dim=7,  # Number of features
        hidden_dim=64,
        num_layers=2,
        output_dim=1
    )
    
    loss_list = train_lstm(lstm_model, X_train_tensor, y_train_tensor)
    
    # 7. SHAP Analysis
    feature_names = ['coastal', 'red', 'green', 'blue', 'nir08', 'swir16', 'NDVI']
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    rf_explainer, rf_shap_values = perform_shap_analysis(rf_model, X_test_df, 'rf')
    xgb_explainer, xgb_shap_values = perform_shap_analysis(xgb_model, X_test_df, 'xgb')
    
    compare_shap_explanations(rf_explainer, xgb_explainer, X_test_df)

if __name__ == "__main__":
    main()