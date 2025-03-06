#model_training.py
import data_processing
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import cudf
import time
import joblib

def train_model(parquet_path = 'data/full_df.parquet', data_proc="no"):
    if data_proc == "yes":
        print("Processing Data")
        full_df = data_processing.process_data()
    else:
        full_df = pd.read_parquet(parquet_path)

    #Split train and test sets

    train_df = full_df

    train_df['e5_volatility'] = train_df['e5'].pct_change().rolling(7).std()
    train_df['e5_lag_1'] = train_df['e5'].shift(1)
    train_df['e5_lag_3'] = train_df['e5'].shift(3)
    train_df['e5_lag_7'] = train_df['e5'].shift(7)
    print(len(train_df))

    #Convert pandas df to cudf for faster gpu training. Deactivate to avoid training with gpu
    #train_df = cudf.from_pandas(train_df)

    #Choose significant columns for training
    feature_cols = ['hour_sin', 'hour_cos', 'station_id_encoded', 'e5_7d_avg', 'oil_price', 'oil_7d_avg', 'e5_volatility',
                    'e5_lag_1', 'e5_lag_3', 'e5_lag_7',]
    #set target column
    target_cols = ['e5']

    print("Splitting Dataframe...")
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=1991)

    train_df = cudf.from_pandas(train_df)
    test_df = cudf.from_pandas(test_df)

    #Create train and test dataframes
    print("Creating train and test dataframes...")
    X_train, y_train = train_df[feature_cols], train_df[target_cols]
    X_test, y_test = test_df[feature_cols], test_df[target_cols]

    batch_size = 300000
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

    #Fitting the model
    print("Fitting model... ")


    for i in range(num_batches):
        print(f"Training batch {i +1} / {num_batches}")
        start_model = time.time()
        batch_df = train_df.iloc[i * batch_size: (i + 1) * batch_size]
        X_batch = batch_df[feature_cols]
        y_batch = batch_df[target_cols]  # ✅ Correct way to slice y_train
        dtrain = xgb.DMatrix(X_batch, label=y_batch)

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
    joblib.dump(xgb_model, 'models/xgbr_trained.pkl')