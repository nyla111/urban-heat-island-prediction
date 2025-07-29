import shap
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from ..config import FEATURE_NAMES


def perform_shap_analysis(model, X_test, model_type='rf'):
    """Perform SHAP analysis on model"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test)
    
    # Bar plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    return explainer, shap_values


def compare_shap_explanations(rf_explainer, xgb_explainer, X_test, instance_idx=1):
    """Compare SHAP explanations between models"""
    shap.initjs()
    
    # Display force plots side-by-side
    from IPython.display import display, HTML
    
    rf_plot = shap.force_plot(
        rf_explainer.expected_value, 
        rf_explainer.shap_values(X_test)[instance_idx], 
        X_test.iloc[instance_idx], 
        matplotlib=False
    )
    
    xgb_plot = shap.force_plot(
        xgb_explainer.expected_value, 
        xgb_explainer.shap_values(X_test)[instance_idx], 
        X_test.iloc[instance_idx], 
        matplotlib=False
    )
    
    display(HTML("<h4>Random Forest Prediction Explanation</h4>"))
    display(rf_plot)
    display(HTML("<h4>XGBoost Prediction Explanation</h4>"))
    display(xgb_plot)