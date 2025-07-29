from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from ..config import TEST_SIZE, RANDOM_STATE


def prepare_train_test(uhi_data):
    """Prepare train/test splits"""
    features_to_keep = ['coastal', 'red', 'green', 'blue', 'nir08', 'swir16', 'NDVI']
    X = uhi_data[features_to_keep].values
    y = uhi_data['UHI'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Scale features
    sc = RobustScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """Train and evaluate Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train and evaluate XGBoost model"""
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    # Training evaluation
    train_pred = model.predict(X_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
    mae_train = mean_absolute_error(y_train, train_pred)
    r2_train = r2_score(y_train, train_pred)
    
    # Testing evaluation
    test_pred = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
    mae_test = mean_absolute_error(y_test, test_pred)
    r2_test = r2_score(y_test, test_pred)
    
    return {
        'cv_rmse': cv_rmse,
        'rmse_train': rmse_train,
        'mae_train': mae_train,
        'r2_train': r2_train,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'r2_test': r2_test
    }