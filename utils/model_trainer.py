"""
utils/model_trainer.py
Trains pre-exam dropout risk model using ONLY demographic features (no leakage).
Real framing: given gender, SES, parent education, test prep → predict at-risk.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.pipeline import Pipeline

from data.load_data import FEATURE_COLS, FEATURE_COLS_POST, TARGET_COL


def make_pipeline(clf):
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def train_all_models(df, use_post_exam=False):
    cols = FEATURE_COLS_POST if use_post_exam else FEATURE_COLS
    X = df[cols].copy()
    y = df[TARGET_COL].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': make_pipeline(
            LogisticRegression(max_iter=1000, random_state=42,
                               class_weight='balanced', C=1.0)
        ),
        'Random Forest': make_pipeline(
            RandomForestClassifier(n_estimators=500, max_depth=5,
                                   min_samples_leaf=5, random_state=42,
                                   class_weight='balanced', n_jobs=-1)
        ),
        'Gradient Boosting': make_pipeline(
            GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, random_state=42,
                                       subsample=0.8)
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        y_pred  = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)[:, 1]
        cv_auc  = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')

        results[name] = {
            'model':       pipe,
            'accuracy':    accuracy_score(y_te, y_pred),
            'precision':   precision_score(y_te, y_pred, zero_division=0),
            'recall':      recall_score(y_te, y_pred, zero_division=0),
            'f1':          f1_score(y_te, y_pred, zero_division=0),
            'roc_auc':     roc_auc_score(y_te, y_proba),
            'cv_auc_mean': cv_auc.mean(),
            'cv_auc_std':  cv_auc.std(),
            'confusion_matrix': confusion_matrix(y_te, y_pred),
            'X_test': X_te, 'y_test': y_te,
            'y_pred': y_pred, 'y_proba': y_proba,
            'feature_cols': cols,
        }

    return results, X_tr, X_te, y_tr, y_te


def get_feature_importance(results, top_n=10):
    best = results['Random Forest']
    clf  = best['model'].named_steps['clf']
    cols = best['feature_cols']
    return (pd.DataFrame({'feature': cols[:len(clf.feature_importances_)],
                           'importance': clf.feature_importances_})
              .sort_values('importance', ascending=False)
              .head(top_n))


def predict_single(model, feature_cols, student_dict):
    row  = pd.DataFrame([{col: student_dict.get(col, 0) for col in feature_cols}])
    prob = model.predict_proba(row)[0][1]
    label = '⚠️ AT RISK' if prob > 0.50 else '✅ ON TRACK'
    return {'probability': prob, 'label': label}
