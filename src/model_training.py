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
    import types
    cudf = types.SimpleNamespace(from_pandas=lambda x: x)
    GPU_ENABLED = False

def load_full_data(parquet_path=paths.FINAL_PARQUET_PATH, process_data=True, fetch=True):
    if process_data:
        print("Processing Oil")
        oil = dp.OilProcessing()
        print("Processing Prices")
        prices = dp.PricesProcessing(fetch=fetch)
        full_df = dp.DataPipeline(oil, prices).process_all()
    else:
        print("Loading final_df.parquet")
        full_df = pd.read_parquet(parquet_path)

    for col in full_df.select_dtypes(include='float').columns:
        full_df[col] = full_df[col].astype('float32')
    for col in full_df.select_dtypes(include='int').columns:
        full_df[col] = full_df[col].astype('int32')

    if 'date' in full_df.columns and isinstance(full_df['date'].dtype, pd.DatetimeTZDtype):
        print("Stripping timezone from 'date' column")
        full_df['date'] = full_df['date'].dt.tz_localize(None)

    print(f"Full DF shape: {full_df.shape}")

    return full_df

def split_train_test(full_df):
    split_index = int(len(full_df) * 0.2)
    train_df = full_df.iloc[:split_index]
    test_df = full_df.iloc[split_index:]

    print(f"Train df length: {len(train_df)}")
    print(f"Test df length: {len(test_df)}")

    if len(train_df) == 0:
        print("⚠️ No training data available. Skipping training.")
        return train_df, test_df  # Return as-is, later code can skip training

    if GPU_ENABLED:
        train_df = cudf.from_pandas(train_df)
        test_df = cudf.from_pandas(test_df)
    return train_df, test_df

def prepare_features(train_df, test_df, feature_cols, target_cols):
    X_train, y_train = train_df[feature_cols], train_df[target_cols]
    X_test, y_test = test_df[feature_cols], test_df[target_cols]
    return X_train, y_train, X_test, y_test

def train_model_batches(X_train, y_train, batch_size=2_000_000):
    num_batches = len(X_train) // batch_size + 1
    print(f"Number of batches: {num_batches}")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        colsample_bytree=0.8,
        max_depth=15,
        min_child_weight=5,
        subsample=0.85,
        gamma=0,
        tree_method="hist",
        device="cuda" if GPU_ENABLED else "auto",
        enable_categorical=True,
        objective="reg:squarederror",
        random_state=1991,
        verbosity=2
    )

    first_batch = True
    for i in range(num_batches):
        print(f"Training batch {i + 1} / {num_batches}")
        start_time = time.time()
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        X_batch = X_train.iloc[batch_slice]
        y_batch = y_train.iloc[batch_slice]

        print("X_batch dtypes:")
        print(X_batch.dtypes)

        if first_batch:
            model.fit(X_batch, y_batch, verbose=True)
            first_batch = False
        else:
            model.fit(X_batch, y_batch, xgb_model=model, verbose=True)
        end_time = time.time()
        print(f"Model training elapsed time for batch {i + 1}: {end_time - start_time:.2f} seconds")

    return model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model on test data...")
    y_pred = model.predict(X_test)
    if GPU_ENABLED:
        mse = mean_squared_error(y_test.to_pandas(), y_pred)
    else:
        mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Data: {mse:.4f}")
    return mse

def save_trained_model(model, model_path=paths.MODEL_PATH):
    print(f"Storing model in {model_path}")
    joblib.dump(model, model_path)

def run_training(fetch_data=False, data_proc=False):
    full_df = load_full_data(fetch=fetch_data, process_data=data_proc)
    train_df, test_df = split_train_test(full_df)

    feature_cols = [
        'hour_sin', 'id', 'weekday_sin',
        'oil_price', 'oil_3d_mean', 'oil_7d_mean', 'post_code'
    ]
    target_cols = ['e5']

    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df, feature_cols, target_cols)
    trained_model = train_model_batches(X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)
    save_trained_model(trained_model)