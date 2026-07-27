"""Точка входа: загрузка данных → предобработка → обучение с CV → сабмит + метрики."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import config
from src.data import load_raw_data, preprocessing
from src.ensemble import average_ensemble, build_test_meta_features, fit_stacking
from src.models import NEEDS_SCALING, build_models
from src.train import run_model


def main():
    seed = config.general.seed
    cv = KFold(n_splits=config.cv.n_splits, shuffle=config.cv.shuffle, random_state=seed)

    train_raw, test_raw = load_raw_data(config.paths.path_to_train, config.paths.path_to_test)
    train, test, train_cat, test_cat, cat_features = preprocessing(train_raw, test_raw, cv)

    X, y = train.drop(columns=["SalePrice"]), train["SalePrice"]
    X_cat, y_cat = train_cat.drop(columns=["SalePrice"]), train_cat["SalePrice"]

    models = build_models(config, seed)

    print("\nMODEL TRAINING:")

    stats, results = {}, {}
    for name, model in models.items():
        X_input, y_input = (X_cat, y_cat) if name == "cb" else (X, y)
        fit_params = {"cat_features": cat_features} if name == "cb" else None
        results[name] = run_model(
            name, model, X_input, y_input, cv,
            needs_scaling=name in NEEDS_SCALING, fit_params=fit_params, stats=stats,
        )

    stacking_members = list(config.training.stacking.meta_members)
    stats["Averaging"] = average_ensemble(results, stacking_members, y)

    meta_model, meta_cv_score = fit_stacking(results, y, cv, config.training.stacking)
    stats["Stacking"] = meta_cv_score

    X_meta_test = build_test_meta_features(results, test, test_cat, stacking_members)
    final_preds_log = meta_model.predict(X_meta_test)
    final_preds = np.expm1(final_preds_log)

    submission = pd.DataFrame({"Id": test_raw["Id"], "SalePrice": final_preds})
    submission.to_csv(config.paths.path_to_submission, index=False)

    with open(config.paths.path_to_metrics, "w") as f:
        json.dump({k: float(v) for k, v in stats.items()}, f, indent=2)

    print(f"\nFINAL TABLE (OOF/CV RMSLE)")
    for name, rmse in sorted(stats.items(), key=lambda x: x[1]):
        print(f"{name:<9}: {rmse:<10.4f}")
    print(f"\nSUBMIT SAVED IN '{config.paths.path_to_submission}'")

if __name__ == "__main__":
    main()