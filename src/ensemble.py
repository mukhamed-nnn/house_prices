"""Усреднение и стекинг поверх OOF-предсказаний базовых моделей."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error


def average_ensemble(results: dict, members: list[str], y) -> float:
    """Простое усреднение OOF-предсказаний нескольких моделей — для сравнения со стекингом."""
    
    oof_avg = np.column_stack([results[name]["oof_preds"] for name in members]).mean(axis=1)
    return root_mean_squared_error(y, oof_avg)


def fit_stacking(results: dict, y, cv, stacking_cfg):
    """Обучает мета-модель (LinearRegression) на честных OOF-предсказаниях базовых моделей.

    Возвращает (обученная мета-модель, честная CV-точность самой мета-модели).
    """
    members = list(stacking_cfg.meta_members)
    X_meta_train = pd.DataFrame({name: results[name]["oof_preds"] for name in members})

    meta_model = LinearRegression(max_iter=1000, C=stacking_cfg.C)
    meta_cv_scores = cross_val_score(meta_model, X_meta_train, y, cv=cv, scoring="root_mean_squared_error")
    meta_model.fit(X_meta_train, y)

    return meta_model, float(meta_cv_scores.mean())


def build_test_meta_features(results: dict, test, test_cat, members: list[str]) -> pd.DataFrame:
    """Строит матрицу мета-признаков для test — предсказания базовых моделей на новых данных."""
    probs = {}
    for name in members:
        model = results[name]["model"]
        scaler = results[name]["scaler"]
        X_input = test_cat if name == "cb" else test
        if scaler is not None:
            X_input = scaler.transform(X_input)
        probs[name] = model.predict_proba(X_input)[:, 1]
    return pd.DataFrame(probs)