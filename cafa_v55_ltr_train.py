#!/usr/bin/env python3
"""
Phase 33: Learning-to-Rank (LTR) - Model Training.
Trains XGBoost to rank GO terms based on multi-modal features.
"""

import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
INPUT_FILE = "ltr_val_dataset.pkl"
MODEL_FILE = "v55_ltr_xgboost.model"

def train_ltr():
    logger.info("Loading Dataset...")
    df = pd.read_pickle(INPUT_FILE)
    
    # Features
    features = ['f_diamond', 'f_sbert', 'f_esm2', 'f_prior']
    X = df[features]
    y = df['label']
    
    # Train/Test Split (within Val set)
    # We split Val into "Train Ranker" and "Test Ranker" to measure AUC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training XGBoost on {len(X_train)} samples...")
    
    # Ranker vs Classifier?
    # Simple Binary Classification (Pointwise LTR) is a good start.
    # It learns P(Correct | Features).
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        n_jobs=-1,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Eval
    logger.info("Evaluating...")
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    logger.info(f"ROC AUC: {auc:.4f}")
    
    # Feature Importance
    logger.info("Feature Importances:")
    imps = model.feature_importances_
    for f, i in zip(features, imps):
        logger.info(f"  {f}: {i:.4f}")
        
    # Save
    logger.info(f"Saving model to {MODEL_FILE}...")
    model.save_model(MODEL_FILE)
    
    # Save Plot?
    # xgb.plot_importance(model)
    # plt.savefig("ltr_importance.png")

if __name__ == "__main__":
    train_ltr()
