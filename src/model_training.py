from src import data_processing as dp
from src import data_fetch as df
import pandas as pd
import pickle
from src.utils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# print("Processing prices")
# prices = dp.PricesProcessing(fetch=False)
# print("Processing oil")
# oil = dp.OilProcessing(fetch=False)
# print("Processing pipeline")
# pipeline = dp.DataPipeline(oil, prices)
# final_df = pipeline.process_all()

final_df = pd.read_parquet(paths.FINAL_PARQUET_PATH)


final_df["brand"] = final_df["brand"].str.strip().str.upper()
final_df["uuid"] = final_df["uuid"].str.strip()


print("Labeling")
le_brand = LabelEncoder()
le_uuid = LabelEncoder()


final_df["brand"] = le_brand.fit_transform(final_df["brand"])
final_df["uuid"] = le_uuid.fit_transform(final_df["uuid"])

with open("encoders.pkl", "wb") as f:
    pickle.dump({"brand": le_brand, "uuid": le_uuid}, f)

change_rows = final_df[final_df['price_change_next_hour'] == 1]
X = change_rows[['diesel', 'brand', 'month', 'day', 'hour', 'diesel_shift']]
y = change_rows["price_change"]
X = X.dropna()
y = y.loc[X.index]
# Create mask from X directly
mask_train = (X['month'] == 3) & (X['day'] <= 20)
mask_test = (X['month'] == 3) & (X['day'] > 20)

X_train = X[mask_train]
X_test = X[mask_test]
if isinstance(X_test, pd.Series):
    X_test = X_test.to_frame()
X_test.to_parquet("X_test.parquet")
y_train = y[mask_train]
y_test = y[mask_test]
if isinstance(y_test, pd.Series):
    y_test = y_test.to_frame()
y_test.to_parquet("y_test.parquet")

print("Training")
import lightgbm as lgb
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'tree_method': 'gpu_hist',
    'verbosity': -1,
    'learning_rate': 0.03,
    'num_leaves': 64,
    'max_depth': 7,
    'class_weight': 'balanced',
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'force_col_wise': True,
}


train_data = lgb.Dataset(X_train.copy(), label=y_train.copy())
valid_data = lgb.Dataset(X_test.copy(), label=y_test.copy())

model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=500,
)
print("Saving model")
model.save_model("lightgbm_price_change_model.txt")

y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred_labels))
print(classification_report(y_test, y_pred_labels))