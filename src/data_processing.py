#data_processing.py
import pandas as pd
import glob
import numpy as np
import os
from src import data_fetch as dfe
from src.utils import paths
from sklearn.preprocessing import LabelEncoder

class OilProcessing:

    def __init__(self, fetch_if_missing=True):
        if not os.path.exists(paths.OIL_CSV_PATH):
            if fetch_if_missing:
                dfe.DataFetcher().fetch_brent()
            else:
                raise FileNotFoundError(f"Oil CSV not found at {paths.OIL_CSV_PATH}")
        self.oil_df = pd.read_csv(paths.OIL_CSV_PATH)

    def set_columns(self):
        self.oil_df['Date'] = pd.to_datetime(self.oil_df['Date'], yearfirst=True)
        self.oil_df['month'] = self.oil_df['Date'].dt.month.astype(np.float32)
        self.oil_df['day'] = self.oil_df['Date'].dt.day.astype(np.float32)
        self.oil_df['year'] = self.oil_df['Date'].dt.year.astype(np.float32)
        self.oil_df.rename(columns={'Close' : 'oil_price'}, inplace=True)
        self.oil_df.drop(columns=['Volume', 'Open', 'High',  'Low', 'Date'], inplace=True)

    def add_rolling_means(self):
        self.oil_df['oil_3d_mean'] = self.oil_df['oil_price'].rolling(3, min_periods=1).mean().astype(np.float32)
        self.oil_df['oil_7d_mean'] = self.oil_df['oil_price'].rolling(7, min_periods=1).mean().astype(np.float32)
        self.oil_df['oil_14d_mean'] = self.oil_df['oil_price'].rolling(14, min_periods=1).mean().astype(np.float32)

    def save_parquet (self, parquet_path=paths.OIL_PARQUET_PATH):
        self.oil_df.to_parquet(parquet_path, index=False)

    def full_processing(self, path=paths.OIL_PARQUET_PATH, store_parquet=True):
        self.set_columns()
        self.add_rolling_means()
        if store_parquet:
            self.save_parquet(parquet_path=path)

class PricesProcessing:

    def __init__(self, gpu=False):
        if not os.path.exists(paths.PRICES_DIR):
            raise FileNotFoundError(f"Prices directory not found at {paths.test_data_path('prices')}")
            #Commented out for testing.
            # raise FileNotFoundError(f"Price files were not found in: {paths.PRICES_DIR}")

        prices_list = glob.glob(os.path.join(paths.test_data_path('prices'), '*', '*-prices.csv'))
        #prices_list = glob.glob(os.path.join(paths.PRICES_DIR, '*', '*-prices.csv'))
        data_frames = []

        if not gpu:
            for price_file in prices_list:
                print(f"Processing {price_file}")
                date_string = os.path.basename(price_file).replace("-prices.csv", "")
                date = pd.to_datetime(date_string, yearfirst=True)
                df = pd.read_csv(price_file, sep=',')
                df['date'] = date
                data_frames.append(df)
                self.full_df = pd.concat(data_frames)

    def set_columns(self):
        #self.full_df['date'] = pd.to_datetime(['date'], yearfirst=True)
        self.full_df['year'] = self.full_df['date'].dt.year
        self.full_df['month'] = self.full_df['date'].dt.month
        self.full_df['day'] = self.full_df['date'].dt.day
        self.full_df['weekday'] = self.full_df['date'].dt.weekday
        self.full_df['hour'] = self.full_df['date'].dt.hour

    def df_cleaning(self):
        encoder = LabelEncoder()
        self.full_df['station_id_encoded'] = encoder.fit_transform(self.full_df['station_uuid'])
        if 'date' in self.full_df.columns:
            self.full_df.drop('date', axis=1, inplace=True)
        self.full_df = self.full_df.dropna()
        self.full_df = self.full_df[(self.full_df['diesel'] >= 0.5) & (self.full_df['diesel'] <= 3)]
        self.full_df = self.full_df[(self.full_df['e5'] >= 0.5) & (self.full_df['e5'] <= 3)]
        self.full_df = self.full_df[(self.full_df['e10'] >= 0.5) & (self.full_df['e10'] <= 3)]


    def set_datetime_sin(self):
        self.full_df['hour_sin'] = np.sin(2 * np.pi * self.full_df['hour'] / 24)
        self.full_df['weekday_sin'] = np.sin(2 * np.pi * self.full_df['weekday'] / 7)

    def set_datetime_cos(self):
        self.full_df['hour_cos'] = np.cos(2 * np.pi * self.full_df['hour'] / 24)
        self.full_df['weekday_cos'] = np.cos(2 * np.pi * self.full_df['weekday'] / 7)

    def save_parquet (self, parquet_path=paths.PRICES_PARQUET_PATH):
        self.full_df.to_parquet(parquet_path, index=False)

    def full_processing(self, path=paths.PRICES_PARQUET_PATH, store=True):
        self.set_columns()
        self.df_cleaning()
        self.set_datetime_sin()
        self.set_datetime_cos()
        if store:
            self.save_parquet(parquet_path=path)

    #TODO gpu is True
    #TODO 3,5,7 day avg, volatility, 1,3,7 day lag
    # df['diesel_7d_avg'] = df['diesel'].rolling(7, min_periods=1).mean()
    # df['e5_7d_avg'] = df['e5'].rolling(7, min_periods=1).mean()
    # df['e10_7d_avg'] = df['e10'].rolling(7, min_periods=1).mean()
    # df['e5_volatility'] = df['e5'].pct_change().rolling(7).std()
    # df['e5_lag_1'] = df['e5'].shift(1)
    # df['e5_lag_3'] = df['e5'].shift(3)
    # df['e5_lag_7'] = df['e5'].shift(7)

class DataPipeline:

    def __init__(self, oil_processor: OilProcessing, prices_processor: PricesProcessing):
        self.oil_processor = oil_processor
        self.prices_processor = prices_processor

    def process_all(self, store=True, final_path=paths.FINAL_PARQUET_PATH) -> pd.DataFrame:
        self.oil_processor.full_processing()
        self.prices_processor.full_processing()
        final_df = self.merge()
        if store:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            final_df.to_parquet(final_path, index=False)
        return final_df

    def merge(self) -> pd.DataFrame:
        return self.prices_processor.full_df.merge(
            self.oil_processor.oil_df, how='left', on=["month", "day", "year"]
        )
