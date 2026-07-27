"""Усреднение и стекинг поверх честных OOF-предсказаний базовых моделей (в лог-пространстве)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


def average_ensemble(results: dict, members: list[str], y) -> float:
    """Простое усреднение OOF-предсказаний нескольких моделей — для сравнения со стекингом."""
    y_log = np.log1p(y)
    oof_avg = np.column_stack([results[name]["oof_preds"] for name in members]).mean(axis=1)
    return float(root_mean_squared_error(y_log, oof_avg))


def fit_stacking(results: dict, y, cv, stacking_cfg):
    """Обучает мета-модель (LinearRegression) на честных OOF-предсказаниях базовых моделей.

    Возвращает (обученная мета-модель, честная CV RMSE(log) самой мета-модели).
    """
    y_log = np.log1p(y)
    members = list(stacking_cfg.meta_members)
    X_meta_train = pd.DataFrame({name: results[name]["oof_preds"] for name in members})

    meta_model = LinearRegression()
    meta_cv_scores = cross_val_score(meta_model, X_meta_train, y_log, cv=cv, scoring="neg_root_mean_squared_error")
    meta_model.fit(X_meta_train, y_log)

    return meta_model, float(-meta_cv_scores.mean())


def build_test_meta_features(results: dict, test, test_cat, members: list[str]) -> pd.DataFrame:
    """Строит матрицу мета-признаков для test — предсказания базовых моделей (в лог-пространстве)."""
    preds = {}
    for name in members:
        model = results[name]["model"]
        scaler = results[name]["scaler"]
        X_input = test_cat if name == "cb" else test
        if scaler is not None:
            X_input = scaler.transform(X_input)
        preds[name] = model.predict(X_input)
    return pd.DataFrame(preds)