#model_training.py
import pandas as pd
import data_processing as dp
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from utils import paths
import time
import joblib
try:
    import cudf
    GPU_ENABLED = True
except ImportError:
    GPU_ENABLED = False

def train_model(parquet_path = paths.FINAL_PARQUET_PATH, data_proc=True):

    if data_proc:
        print("Processing Data")
        oil = dp.OilProcessing().full_processing(store_parquet=False)
        prices = dp.PricesProcessing(gpu = GPU_ENABLED).full_processing(store_parquet=False)
        full_df = dp.DataPipeline(oil, prices).process_all()
    else:
        full_df = pd.read_parquet(parquet_path)

    if GPU_ENABLED:
        full_df = cudf.from_pandas(full_df)

    #Choose significant columns for training
    feature_cols = ['hour_sin', 'hour_cos', 'station_id_encoded', 'e5_7d_avg', 'oil_price', 'oil_7d_avg', 'e5_volatility',
                    'e5_lag_1', 'e5_lag_3', 'e5_lag_7']
    #set target column
    target_cols = ['e5']

    max_date = full_df['date'].max()
    test_start = max_date - pd.Timedelta(days=10)

    print("Splitting Dataframe...")
    train_df = full_df[full_df['date'] < test_start]
    test_df = full_df[full_df['date'] >= test_start]

    #Create train and test dataframes
    print("Creating train and test dataframes...")
    X_train, y_train = train_df[feature_cols], train_df[target_cols]
    X_test, y_test = test_df[feature_cols], test_df[target_cols]

    batch_size = 2000000            #Adapt batch_size according to available Hardware
    num_batches = len(X_train) // batch_size + 1
    print(f"Number of batches: {num_batches}")
    first_batch = True
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        colsample_bytree=0.8,
        max_depth=15,
        min_child_weight=5,
        subsample=0.85,
        gamma=0,
        tree_method="hist",
        device="cuda",                  #Remove if training on CPU
        enable_categorical=True,
        objective="reg:squarederror",   #Supports training multiple targets
        random_state=1991,
        verbosity = 2
    )

    for i in range(num_batches):
        print(f"Training batch {i +1} / {num_batches}")
        start_model = time.time()
        batch_df = train_df.iloc[i * batch_size: (i + 1) * batch_size]
        X_batch = batch_df[feature_cols]
        y_batch = batch_df[target_cols]

        if first_batch:
            xgb_model.fit(X_batch, y_batch, verbose=True)
            first_batch = False
        else:
            xgb_model.fit(X_batch, y_batch, xgb_model=xgb_model, verbose=True)
        end_model = time.time()
        print("Model training elapsed time: {:.2f} seconds".format(end_model - start_model))

    print("Evaluating model on test data...")
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test.to_pandas(), y_pred)  # Convert cudf to pandas for sklearn
    print(f"Mean Squared Error on Test Data: {mse:.4f}")

    #Store the trained model
    print("Storing model in models/xgbr_trained.pkl")
    joblib.dump(xgb_model, '../models/xgbr_trained.pkl')