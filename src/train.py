"""Честная OOF-оценка модели и построение финальной модели на 100% данных."""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler


def run_model(name, estimator, X, y, cv, needs_scaling=False, fit_params=None, stats=None):
    """ Честная OOF-оценка + финальная модель на 100% данных. """
    fit_params = fit_params or {}
    y_log = np.log1p(y)
    oof_preds = np.zeros(len(y))
    fold_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

        if needs_scaling:
            scaler = StandardScaler()
            X_tr, X_val = scaler.fit_transform(X_tr), scaler.transform(X_val)

        model = clone(estimator)
        model.fit(X_tr, y_tr, **fit_params)

        fold_pred = model.predict(X_val)
        oof_preds[val_idx] = fold_pred
        fold_scores.append(root_mean_squared_error(y_val, fold_pred))

    total_rmse = root_mean_squared_error(y_log, oof_preds)
    if stats is not None:
        stats[name] = total_rmse

    msg = f"{name:<5}: OOF RMSLE = {total_rmse:.4f} │ by folds: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}"
    print(msg)

    scaler_full = StandardScaler().fit(X) if needs_scaling else None
    X_full = scaler_full.transform(X) if scaler_full else X
    final_model = clone(estimator).fit(X_full, y_log, **fit_params)

    return {
        'oof_preds': oof_preds, 'model': final_model, 'scaler': scaler_full, 'fold_scores': fold_scores}