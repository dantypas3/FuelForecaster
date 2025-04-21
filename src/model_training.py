import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from src import data_processing as dp
from src.utils import paths

def create_train_df(fetch_data=False, load_parquet=False):
    print("Processing Data")
    if load_parquet:
        final_df = pd.read_parquet(paths.FINAL_PARQUET_PATH)
    else:
        prices   = dp.PricesProcessing(fetch=fetch_data)
        pipeline = dp.DataPipeline(prices)
        final_df = pipeline.process_all()

    df = final_df.copy()
    df = df.sort_values(['month','day']).dropna(subset=['diesel','diesel_shift','price_will_change'])
    df.drop(columns=['change', 'e5', 'e10'], inplace=True)
    return df

def prepare_cols(df):
    print("Encoding features")
    df['brand'] = df['brand'].str.strip().str.upper()
    df['uuid']  = df['uuid'].str.strip()
    le_brand = LabelEncoder()
    le_uuid = LabelEncoder()
    df['brand'] = le_brand.fit_transform(df['brand'])
    df['uuid']  = le_uuid.fit_transform(df['uuid'])

    with open(paths.ENCODERS_PATH,"wb") as f:
        pickle.dump({"brand":le_brand,"uuid":le_uuid}, f)

    return le_brand, le_uuid

def split_train_test(df):
    df["day_idx"] = (df['month'].astype(str).str.zfill(2) + df['day'].astype(str).str.zfill(2)).astype(int)
    df["day_idx"] = df["day_idx"].rank(method="dense").astype(int)
    max_day = df["day_idx"].max()
    df = df[df["day_idx"] > max_day - 30].reset_index(drop=True)

    # 5) Build train/test splits (first 35 days vs last 10)
    feature_cols = ['uuid', 'diesel','day','hour', 'brand', 'diesel_shift']
    X = df[feature_cols]
    y = df['price_will_change']
    cutoff = df["day_idx"].min() + 20
    mask_train = df["day_idx"] <= cutoff
    X_train, y_train = X[mask_train], y[mask_train]
    X_test,  y_test  = X[~mask_train], y[~mask_train]

    df.to_parquet(paths.data_path("parquets","train_filtered_df.parquet"), index=False)

    X_test_with_idx = X_test.copy()
    X_test_with_idx["orig_idx"] = X_test.index
    y_test_with_idx = (
        y_test
        .reset_index()
        .rename(columns={'index':'orig_idx','price_will_change':'y_true'})
    )

    X_test_with_idx.to_parquet(paths.data_path("parquets","X_test.parquet"), index=False)
    y_test_with_idx.to_parquet(paths.data_path("parquets","y_test.parquet"), index=False)

    return X_train, X_test, y_train, y_test


def train_diesel_change(fetch_data=False, load_parquet=False):

    df = create_train_df(fetch_data, load_parquet)
    prepare_cols(df)
    X_train, X_test, y_train, y_test = split_train_test(df)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test,  label=y_test)

    params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'boosting_type': 'gbdt', 'device': 'gpu', 'verbosity': -1,
        'num_leaves': 64, 'min_child_samples': 200, 'max_depth': -1,
        'learning_rate': 0.1, 'lambda_l1': 0.5, 'lambda_l2': 0.5,
        'feature_fraction': 0.9, 'bagging_freq': 1, 'bagging_fraction': 0.9
    }

    print("Training Model")
    model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], num_boost_round=500)
    print("Saving Model")
    model.save_model(paths.MODEL_PATH)