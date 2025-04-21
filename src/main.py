# main.py

import argparse
from src import model_training as mt
from evaluate_diesel_train import evaluate_model, tune_hyperparams

def build_parser():
    p = argparse.ArgumentParser(
        description="Manage diesel‑change model: train, eval, or tune"
    )
    subs = p.add_subparsers(dest="command", required=True)

    # ─── train subcommand ──────────────────────────────────────────────
    train = subs.add_parser("train", help="Fetch/process data and train model")
    train.add_argument(
        "--fetch", action="store_true",
        help="Fetch the latest price data"
    )
    train.add_argument(
        "--load-parquet", action="store_true",
        help="Load existing parquet instead of re-fetching"
    )

    # ─── eval subcommand ───────────────────────────────────────────────
    eval_ = subs.add_parser("eval", help="Load saved model and evaluate on X_test/y_test")
    eval_.add_argument(
        "--threshold", type=float, default=0.59,
        help="Cutoff for turning probabilities into class labels"
    )

    # ─── tune subcommand ───────────────────────────────────────────────
    tune = subs.add_parser("tune", help="Run RandomizedSearchCV to find best hyperparams")
    tune.add_argument(
        "--train-days", type=int, default=20,
        help="Number of days for training (rest for validation)"
    )
    tune.add_argument(
        "--n-iter", type=int, default=50,
        help="How many random‑search trials"
    )
    tune.add_argument(
        "--cv-folds", type=int, default=4,
        help="Number of CV folds"
    )
    tune.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel jobs for search"
    )
    tune.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility"
    )
    tune.add_argument(
            "--log-file", type=str, default="tune_results.txt",
            help = "Path to save tuning logs"
    )

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        mt.train_diesel_change(
            fetch_data=args.fetch,
            load_parquet=args.load_parquet,
        )

    elif args.command == "eval":
        evaluate_model(threshold=args.threshold)

    elif args.command == "tune":
        tune_hyperparams(
            train_days=args.train_days,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
            random_state=args.random_state
        )

if __name__ == "__main__":
    main()
