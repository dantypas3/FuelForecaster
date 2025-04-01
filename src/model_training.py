import argparse
import time
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from utils import paths
import data_processing as dp

try:
    import cudf
    GPU_ENABLED = True
except ImportError:
    GPU_ENABLED = False


def load_full_data(parquet_path=paths.FINAL_PARQUET_PATH, process_data=True):
    if process_data:
        print("Processing Data")
        oil = dp.OilProcessing()E
        oil.full_processing(store_parquet=False)
        prices = dp.PricesProcessing(gpu=GPU_ENABLED)
        prices.full_processing(store_parquet=False)
        full_df = dp.DataPipeline(oil, prices).process_all()
    else:
        full_df = pd.read_parquet(parquet_path)

    # Convert to cudf if GPU is available
    if GPU_ENABLED:
        full_df = cudf.from_pandas(full_df)
    return full_df


def split_train_test(full_df):
    max_date = full_df['date'].max()
    test_start = max_date - pd.Timedelta(days=10)
    print("Splitting Dataframe based on date...")
    train_df = full_df[full_df['date'] < test_start]
    test_df = full_df[full_df['date'] >= test_start]
    print(f"Train df length: {len(train_df)}")
    print(f"Test df length: {len(test_df)}")
    return train_df, test_df


def prepare_features(train_df, test_df, feature_cols, target_cols):
    X_train, y_train = train_df[feature_cols], train_df[target_cols]
    X_test, y_test = test_df[feature_cols], test_df[target_cols]
    return X_train, y_train, X_test, y_test


def train_model_batches(X_train, y_train, feature_cols, batch_size=2000000):
    num_batches = len(X_train) // batch_size + 1
    print(f"Number of batches: {num_batches}")

    if GPU_ENABLED:
    # Initialize XGBRegressor with hyperparameters.
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            colsample_bytree=0.8,
            max_depth=15,
            min_child_weight=5,
            subsample=0.85,
            gamma=0,
            tree_method="hist",
            device="cuda",
            enable_categorical=True,
            objective="reg:squarederror",
            random_state=1991,
            verbosity=2
        )
    else:
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            colsample_bytree=0.8,
            max_depth=15,
            min_child_weight=5,
            subsample=0.85,
            gamma=0,
            tree_method="hist",
            enable_categorical=True,
            objective="reg:squarederror",
            random_state=1991,
            verbosity=2
        )

    first_batch = True
    for i in range(num_batches):
        print(f"Training batch {i + 1} / {num_batches}")
        start_time = time.time()
        # Using iloc to slice batches from the training train_data.
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        X_batch = X_train.iloc[batch_slice]
        y_batch = y_train.iloc[batch_slice]
        if first_batch:
            model.fit(X_batch, y_batch, verbose=True)
            first_batch = False
        else:
            # Continue training by passing the already trained model via the xgb_model parameter.
            model.fit(X_batch, y_batch, xgb_model=model, verbose=True)
        end_time = time.time()
        print("Model training elapsed time for batch {}: {:.2f} seconds".format(i + 1, end_time - start_time))
    return model


def evaluate_model(model, X_test, y_test):
    print("Evaluating model on test train_data...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test.to_pandas(), y_pred)  # convert cudf to pandas if necessary
    print(f"Mean Squared Error on Test Data: {mse:.4f}")
    return mse


def save_trained_model(model, model_path=paths.MODEL_PATH):
    print(f"Storing model in {model_path}")
    joblib.dump(model, model_path)


def run_training(data_proc=True):
    """
    Main training pipeline:
      1. Load the full train_data.
      2. Split into train and test sets.
      3. Prepare features.
      4. Train the model in batches.
      5. Evaluate the model.
      6. Save the model.
    """
    full_df = load_full_data(process_data=data_proc)
    train_df, test_df = split_train_test(full_df)

    feature_cols = [
        'hour_sin', 'hour_cos', 'station_id_encoded', 'e5_7d_avg', 
        'oil_price', 'oil_7d_avg', 'e5_volatility', 'e5_lag_1', 
        'e5_lag_3', 'e5_lag_7'
    ]
    target_cols = ['e5']

    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df, feature_cols, target_cols)
    trained_model = train_model_batches(X_train, y_train, feature_cols)
    evaluate_model(trained_model, X_test, y_test)
    save_trained_model(trained_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--process-train_data',
        action='store_true',
        help='Process raw train_data instead of reading from parquet.'
    )
    args = parser.parse_args()

    run_training(data_proc=args.process_data)

if __name__ == '__main__':
    main()