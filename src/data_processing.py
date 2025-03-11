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

class StationProcessing:
    pass
    #TODO all variables 32bit
    #TODO StationsProcessing


def process_data():
    prices_list = glob.glob('data/prices/*/*')
    data_frames = []

    for price_file in prices_list:

        print(f"Processing {price_file}")
        #Read CSV
        df = pd.read_csv(price_file)
        #Convert date column to datetime format
        df['date'] = pd.to_datetime(df['date'], yearfirst=True)

        #Add date-related columns
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['hour'] = df['date'].dt.hour

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)

        df['diesel_7d_avg'] = df['diesel'].rolling(7, min_periods=1).mean()
        df['e5_7d_avg'] = df['e5'].rolling(7, min_periods=1).mean()
        df['e10_7d_avg'] = df['e10'].rolling(7, min_periods=1).mean()
        df['e5_volatility'] = df['e5'].pct_change().rolling(7).std()
        df['e5_lag_1'] = df['e5'].shift(1)
        df['e5_lag_3'] = df['e5'].shift(3)
        df['e5_lag_7'] = df['e5'].shift(7)


        #Convert price to 32bit floats
        df['diesel'] = df['diesel'].astype("float32")
        df['e5'] = df['e5'].astype("float32")
        df['e10'] = df['e10'].astype("float32")
        df['hour_sin'] = df['hour_sin'].astype("float32")
        df['hour_cos'] = df['hour_cos'].astype("float32")
        df['weekday_sin'] = df['weekday_sin'].astype("float32")
        df['weekday_cos'] = df['weekday_cos'].astype("float32")
        df['day_sin'] = df['day_sin'].astype("float32")
        df['day_cos'] = df['day_cos'].astype("float32")
        df['diesel_7d_avg'] = df['diesel_7d_avg'].astype("float32")
        df['e5_7d_avg'] = df['e5_7d_avg'].astype("float32")
        df['e10_7d_avg'] = df['e10_7d_avg'].astype("float32")

        #Store df in data_frames list
        data_frames.append(df)

    #Concatenate all dataframes
    print("Concatenating data frames...")
    full_df = pd.concat(data_frames)

    #oil_df = pd.read_parquet('data/oil_df.parquet')
    #full_df = pd.read_parquet('data/conc_dfs.parquet')
    #Make station id readable by ML model with encoder
    encoder = LabelEncoder()
    full_df['station_id_encoded'] = encoder.fit_transform(full_df['station_uuid'])

    print("Storing concatenated dfs in data/parquets//conc_dfs.parquet")
    full_df.to_parquet('../data/parquets//conc_dfs.parquet', index=False)

    print("full_df columns:", full_df.columns)



    print("Merging Dataframes ...")
    full_df = full_df.merge(oil_df, on=["month", "day", "year"], how='left')


    if full_df['oil_price'].isna().sum() > 0:
        print(f"NaN oil prices before filling: {full_df['oil_price'].isna().sum()}")

        full_df['oil_price'] = full_df['oil_price'].where(full_df['oil_price'].notna(), full_df['oil_price'].ffill())
        full_df['oil_price'] = full_df['oil_price'].where(full_df['oil_price'].notna(), full_df['oil_price'].bfill())
        full_df['oil_price'] = full_df['oil_price'].rolling(7, min_periods=1).mean()
        print(f"NaN oil prices after filling: {full_df['oil_price'].isna().sum()}")

#Clean dataframe
    print("Cleaning Dataframe... ")
    # Free memory by removing date column
    full_df.drop(columns=['date', 'dieselchange', 'e5change', 'e10change'], inplace=True)
    full_df = full_df.dropna()
    full_df = full_df[(full_df['diesel'] >= 0.5) & (full_df['diesel'] <= 3)]
    full_df = full_df[(full_df['e5'] >= 0.5) & (full_df['e5'] <= 3)]
    full_df = full_df[(full_df['e10'] >= 0.5) & (full_df['e10'] <= 3)]
    print(f"Months & Years in df: {full_df['month'].unique()}, {full_df['year'].unique()}")

    print("Storing Dataframe in data/parquets//full_df.parquet")
    full_df.to_parquet('../data/parquets//full_df.parquet', index=False)
    return full_df