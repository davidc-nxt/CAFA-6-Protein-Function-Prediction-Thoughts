#!/usr/bin/env python3
"""
LTR Hyperparameter Tuning for V56.
Optimizes XGBoost parameters for better F-max.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
INPUT_FILE = "ltr_val_dataset.pkl"
MODEL_FILE = "v56_ltr_xgboost_tuned.model"

def tune_ltr():
    logger.info("Loading Dataset...")
    df = pd.read_pickle(INPUT_FILE)
    
    # Features
    features = ['f_diamond', 'f_sbert', 'f_esm2', 'f_prior']
    X = df[features]
    y = df['label']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training data: {len(X_train)} samples")
    logger.info(f"Test data: {len(X_test)} samples")
    
    # Parameter Grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    
    # Base Model
    base_model = xgb.XGBClassifier(
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    
    # Grid Search with AUC
    logger.info("Starting Grid Search...")
    
    # Use smaller subset for faster search
    sample_size = min(100000, len(X_train))
    X_sample = X_train.sample(n=sample_size, random_state=42)
    y_sample = y_train.loc[X_sample.index]
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_sample, y_sample)
    
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    logger.info(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    # Train final model with best params on full data
    logger.info("Training final model...")
    best_model = xgb.XGBClassifier(**grid_search.best_params_, eval_metric='logloss', n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    # Evaluate
    preds = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    logger.info(f"Test AUC: {auc:.4f}")
    
    # Feature Importance
    logger.info("Feature Importances:")
    imps = best_model.feature_importances_
    for f, i in zip(features, imps):
        logger.info(f"  {f}: {i:.4f}")
        
    # Save
    logger.info(f"Saving to {MODEL_FILE}...")
    best_model.save_model(MODEL_FILE)
    
    return grid_search.best_params_

if __name__ == "__main__":
    tune_ltr()
