#data_processing.py
import pandas as pd
import glob
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

class OilProcessing:

    def __init__(self, csv_oil_path):
        if not csv_oil_path:
            raise ValueError("Path for oil.csv was not provided")
        if not os.path.exists(csv_oil_path):
            raise FileNotFoundError(f"File not found: {csv_oil_path}")

        self.oil_df = pd.read_csv(csv_oil_path, sep=',')
        self.set_columns()
        self.df_cleaning()
        self.add_rolling_means()

    def set_columns(self):
        print("Actual columns in DataFrame:", self.oil_df.columns)  # Debugging step
        if not self.oil_df.columns.str.contains('Europe Brent Spot Price FOB  Dollars per Barrel').any():
            raise Exception("Europe Brent Spot Price FOB  Dollars per Barrel' not found")
        self.oil_df.rename(columns={"Europe Brent Spot Price FOB  Dollars per Barrel": "oil_price"}, inplace=True)
        self.oil_df['Day'] = pd.to_datetime(self.oil_df['Day'], yearfirst=True)
        self.oil_df['month'] = self.oil_df['Day'].dt.month
        self.oil_df['day'] = self.oil_df['Day'].dt.day
        self.oil_df['year'] = self.oil_df['Day'].dt.year
        print("Actual columns in DataFrame:", self.oil_df.columns)  # Debugging step


    def df_cleaning(self):
        self.oil_df.drop(columns=['Day'], inplace=True)

    def add_rolling_means(self):
        self.oil_df['oil_7d_mean'] = self.oil_df['oil_price'].rolling(3, min_periods=1).mean()
        self.oil_df['oil_7d_mean'] = self.oil_df['oil_price'].rolling(7, min_periods=1).mean()
        self.oil_df['oil_14d_mean'] = self.oil_df['oil_price'].rolling(14, min_periods=1).mean()


    def save_parquet (self, parquet_name='../data/parquets/oil_df.parquet'):
        self.oil_df.to_parquet(parquet_name, index=False)

    oil_df = None

class StationProcessing:
    def __init__(self, directory_prices_path, gpu=False):
        if not directory_prices_path:
            raise ValueError("Path for station.csv was not provided")
        if not os.path.exists(directory_prices_path):
            raise FileNotFoundError(f"Price files were not found in: {directory_prices_path}")

        prices_list = glob.glob(os.path.join(directory_prices_path, '*', '*'))
        data_frames = []

        if gpu is False:
            for price_file in prices_list:
                print(f"Processing {price_file}")
                df = pd.read_csv(price_file, sep=',')
                data_frames.append(df)
                self.full_df = pd.concat(data_frames)
        self.set_datetime()
        self.set_datetime_sin()
        #self.set_datetime_cos()
        self.df_cleaning()
        self.save_parquet()

    def set_datetime(self):
        self.full_df['date'] = pd.to_datetime(['date'], yearfirst=True)
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

    def save_parquet (self):
        self.full_df.to_parquet('../data/parquets//full_df.parquet', index=False)

    #TODO gpu is True
    #TODO 3,5,7 day avg, volatility, 1,3,7 day lag
    # df['diesel_7d_avg'] = df['diesel'].rolling(7, min_periods=1).mean()
    # df['e5_7d_avg'] = df['e5'].rolling(7, min_periods=1).mean()
    # df['e10_7d_avg'] = df['e10'].rolling(7, min_periods=1).mean()
    # df['e5_volatility'] = df['e5'].pct_change().rolling(7).std()
    # df['e5_lag_1'] = df['e5'].shift(1)
    # df['e5_lag_3'] = df['e5'].shift(3)
    # df['e5_lag_7'] = df['e5'].shift(7)

def process_data_csv(oil: OilProcessing, stations: StationProcessing, save = True) -> pd.DataFrame:
    final_df = stations.full_df.merge(oil, how='left', on=["month", "day", "year"])
    if save:
        final_df.to_parquet('../data/parquets//final_df.parquet', index=False)
    return final_df

def process_data_parquet(oil_parquet = '../data/parquets/oil_df.parquet', stations_parquet = '../data/parquets//full_df.parquet',
                         save = True) -> pd.DataFrame:
    oil = pd.read.parquet(oil_parquet)
    stations = pd.read_parquet(stations_parquet)
    final_df = stations.full_df.merge(oil, how='left', on=["month", "day", "year"])
    if save:
        final_df.to_parquet('../data/parquets//final_df.parquet', index=False)
    return final_df