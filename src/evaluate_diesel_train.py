#evaluat_model.py

import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from joblib import parallel_backend
from lightgbm import LGBMClassifier
from datetime import datetime

from src.utils.paths import data_path, model_path

FEATURE_COLS = ['uuid', 'diesel', 'day', 'hour', 'brand', 'diesel_shift']
TRAIN_FILTERED_PATH = data_path("parquets", "train_filtered_df.parquet")
X_TEST_PATH = data_path("parquets", "X_test.parquet")
Y_TEST_PATH = data_path("parquets", "y_test.parquet")
MODEL_PATH = model_path("lightgbm_price_change_model.txt")
LOG_PATH = data_path("logfiles", "tune_results.txt")

def load_test():

	X_test = pd.read_parquet(X_TEST_PATH)
	y_df = pd.read_parquet(Y_TEST_PATH)
	df = X_test.merge(y_df, on="orig_idx")
	return df[FEATURE_COLS], df['y_true']

def load_train_valid(train_days: int):
	df = pd.read_parquet(TRAIN_FILTERED_PATH)
	cutoff = df["day_idx"].min() + train_days
	train_df = df[df["day_idx"] <= cutoff]
	valid_df = df[df["day_idx"] > cutoff]

	X_train = train_df[FEATURE_COLS]
	y_train = train_df['price_will_change']
	X_valid = valid_df[FEATURE_COLS]
	y_valid = valid_df['price_will_change']

	return X_train, y_train, X_valid, y_valid

def evaluate_model(threshold: float):
    print(f"\nLoading model from `{MODEL_PATH}`")
    model = lgb.Booster(model_file=MODEL_PATH)

    X_test, y_test = load_test()
    y_prob = model.predict(X_test)
    y_pred = (y_prob > threshold).astype(int)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, digits=4)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n=== Confusion Matrix ===")
    print(conf_matrix)
    print("\n=== Classification Report ===")
    print(class_report)
    print(f"\nROC AUC: {roc_auc:.4f}")

    with open(data_path("logfiles", "model_eval.txt"), "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report + "\n")
        f.write(f"\nROC AUC: {roc_auc:.4f}\n")
        f.write("=" * 50 + "\n")

def log_tune_results(score, best_params, run_info, log_file=LOG_PATH):
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for key, val in run_info.items():
            f.write(f"{key}: {val}\n")
        f.write(f"Best CV logloss: {score:.6f}\n")
        f.write("Best Parameters:\n")
        for k, v in best_params.items():
            f.write(f"  • {k}: {v}\n")
        f.write("=" * 50 + "\n")


def tune_hyperparams(
    train_days: int,
    n_iter: int,
    cv_folds: int,
    n_jobs: int,
    random_state: int
):
    print("\nLoading train/valid split for hyperparameter search")
    X_train, y_train, X_valid, y_valid = load_train_valid(train_days)

    clf = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1,
        class_weight='balanced',
        n_estimators=100
    )

    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 64],
        'max_depth': [-1, 5, 7],
        'min_child_samples': [100, 200],
        'feature_fraction': [0.7, 0.9],
        'bagging_fraction': [0.7, 0.9],
        'bagging_freq': [1],
        'lambda_l1': [0, 0.5],
        'lambda_l2': [0, 0.5],
    }

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_log_loss',
        cv=cv_folds,
        verbose=2,
        random_state=random_state,
        n_jobs=n_jobs
    )

    with parallel_backend("threading"):
        search.fit(
            X_train, y_train,
            eval_metric='binary_logloss',
            eval_set=[(X_valid, y_valid)],
        )

    best_score = -search.best_score_
    best_params = search.best_params_

    print("\nModel params after training:")
    print(search.best_estimator_.get_params())
    print(f"\nBest CV logloss: {best_score:.6f}")
    print("Best hyperparameters:")
    for param, val in best_params.items():
        print(f"  • {param}: {val}")

    log_tune_results(
        score=best_score,
        best_params=best_params,
        run_info={
            "train_days": train_days,
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "n_jobs": n_jobs,
            "random_state": random_state
        },
        log_file= data_path("logfiles", "tune_results.txt")
    )

