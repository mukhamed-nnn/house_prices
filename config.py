import torch
from omegaconf import OmegaConf

config = {
    "general": {
        "experiment_name": "house_prices_baseline_v1",
        "seed": 0xDEF,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "paths": {
        "path_to_train": "./data/train.csv",
        "path_to_test": "./data/test.csv",
        "path_to_submission": "./output/submission.csv",
        "path_to_metrics": "./output/metrics.json",
    },
    "cv": {
        "n_splits": 5,
        "shuffle": True,
    },
    "training": {
        "dummy": {
            "strategy": "mean",
        },
        "lr": {
            "alpha": 0.005179,
            "l1_ratio": 0.900000,
            "max_iter": 20000,
            "tol": 1e-3,
        },
        "knn": {
            "n_neighbors": 7,
            "weights": "distance",
            "metric": "manhattan",
        },
        "decision_tree": {
            "criterion": "absolute_error",
            "max_depth": 10,
            "min_samples_split": 16,
            "min_samples_leaf": 13,
            "max_features": None,
        },
        "random_forest": {
            "n_estimators": 300,
            "criterion": "squared_error",
            "max_samples": 0.7,
            "max_depth": 5,
            "max_features": 0.4,
            "min_samples_split": 7,
            "min_samples_leaf": 4,
        },
        "catboost": {
            "depth": 3,
            "iterations": 300,
            "loss_function": "RMSE",
            "learning_rate": 0.05,
            "l2_leaf_reg": 1,
            "random_strength": 0.0,
            "bagging_temperature": 1.0,
        },
        "lightgbm": {
            "n_estimators": 200,
            "learning_rate": 0.0775,
            "num_leaves": 5,
            "max_depth": 7,
            "min_child_samples": 15,
        },
        "xgboost": {
            "n_estimators": 200,
            "learning_rate": 0.055,
            "max_depth": 4,
            "min_child_weight": 4,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
        },
        "nn": {
            "hidden_dim1": 56, 
            "hidden_dim2": 32, 
            "hidden_dim3": 16,
            "dropout1": 0.3, 
            "dropout2": 0.2, 
            "dropout3": 0.1,
            "num_epochs": 150,
            "lr": 0.005,
            "weight_decay": 1e-4,
            "early_stopping": {"patience": 50, "min_delta": 1e-4},
            "scheduler": {"patience": 10, "factor": 0.1},
        },
        "stacking": {
            "meta_members": ["lr", "lgbm", "xgb", "cb"],
        },
    },
    "dataloader": {"batch_size": 32, "num_workers": 2, "shuffle": True},
}

config = OmegaConf.create(config)