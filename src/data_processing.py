# data_processing.py
import pandas as pd
import numpy as np
import os
from src import data_fetch as dfe
from src.utils import paths

class PricesProcessing:
	def __init__(self, gpu=False, fetch=False):
		if fetch:
			dfe.StationFetcher().fetch_all()

		if not gpu:
			station_prices = pd.read_csv(paths.STATION_PRICES_CSV, sep="\t")
			station_data = pd.read_csv(paths.STATION_DATA_CSV, sep="\t")

			station_data.drop(columns=[
				'version', 'version_time', 'name', 'street', 'house_number', 'place', 'price_in_import',
				'price_changed', 'open_ts', 'ot_json', 'station_in_import', 'first_active'
			], inplace=True, errors='ignore')

			self.full_df = pd.merge(station_prices, station_data, how='left', left_on='uuid', right_on='id')
			self.full_df = self.full_df.drop(columns=['id_y']).rename(columns={'id_x': 'id'})
			self.full_df['post_code'] = self.full_df['post_code'].replace('\\N', np.nan)
			self.full_df['post_code'] = self.full_df['post_code'].astype('float').astype(
					'Int32')
		else:
			raise NotImplementedError("GPU mode is not implemented yet.")

	def set_columns(self):

		self.full_df['date'] = pd.to_datetime(self.full_df['date'], yearfirst=True, utc=True)
		self.full_df['date'] = self.full_df['date'].dt.tz_localize(None)
		self.full_df['hour'] = self.full_df['date'].dt.hour
		self.full_df['minute'] = self.full_df['date'].dt.minute
		self.full_df['month'] = self.full_df['date'].dt.month
		self.full_df['day'] = self.full_df['date'].dt.day
		self.full_df['weekday'] = self.full_df['date'].dt.weekday
		self.full_df['hour_sin'] = np.sin(2 * np.pi * self.full_df['hour'] / 24)
		self.full_df['weekday_sin'] = np.sin(2 * np.pi * self.full_df['weekday'] / 7)

		price_columns = ['e5', 'e10', 'diesel']     #Normalize Prices
		for col in price_columns:
			self.full_df[col] = self.full_df[col].where(self.full_df[col] >= 0).astype(float) / 1000

		self.full_df = self.full_df.sort_values(['uuid', 'date'])
		self.full_df['diesel_shift'] = self.full_df.groupby('uuid')['diesel'].shift(-1)
		self.full_df['price_will_change'] = (self.full_df['diesel'] != self.full_df['diesel_shift']).astype(int)

		self.full_df.drop(columns=[
			'change', 'public_holiday_identifier', 'weekday', 'post_code', 'lat', 'lng'
		], inplace=True, errors='ignore')

	def compute_means(self):    #This method will be used in the future for price predictions
		diesel_mean = (self.full_df[self.full_df[["uuid", "month", "day", "diesel"]]["diesel"] > 1]
		               .groupby(["uuid", "month", "day"], as_index=False)
		               .agg(diesel_1d_mean=("diesel", "mean"))
		               .sort_values(by=["uuid", "month", "day"])
		               .assign(**{
			"diesel_3d_mean": lambda df: (df.groupby("uuid")["diesel_1d_mean"]
			                              .rolling(window=3, min_periods=1)
			                              .mean()
			                              .astype(np.float32)
			                              .reset_index(level=0, drop=True)),

			"diesel_7d_mean": lambda df: df.groupby("uuid")["diesel_1d_mean"]
			                   .rolling(window=7, min_periods=1)
			                   .mean()
			                   .reset_index(level=0, drop=True)})
		               )

		e5_mean = (self.full_df[self.full_df[["uuid", "month", "day", "e5"]]["e5"] > 1]
		           .groupby(["uuid", "month", "day"], as_index=False)
		           .agg(e5_1d_mean=("e5", "mean"))
		           .sort_values(by=["uuid", "month", "day"])
		           .assign(**{
			"e5_3d_mean": lambda df: (df.groupby("uuid")["e5_1d_mean"]
			                          .rolling(window=3, min_periods=1)
			                          .mean()
			                          .astype(np.float32)
			                          .reset_index(level=0, drop=True)),

			"e5_7d_mean": lambda df: df.groupby("uuid")["e5_1d_mean"]
			               .rolling(window=7, min_periods=1)
			               .mean()
			               .reset_index(level=0, drop=True)})
		           )

		e10_mean = (self.full_df[self.full_df[["uuid", "month", "day", "e10"]]["e10"] > 1]
		            .groupby(["uuid", "month", "day"], as_index=False)
		            .agg(e10_1d_mean=("e10", "mean"))
		            .sort_values(by=["uuid", "month", "day"])
		            .assign(**{
			"e10_3d_mean": lambda df: (df.groupby("uuid")["e10_1d_mean"]
			                           .rolling(window=3, min_periods=1)
			                           .mean()
			                           .astype(np.float32)
			                           .reset_index(level=0, drop=True)),

			"e10_7d_mean": lambda df: df.groupby("uuid")["e10_1d_mean"]
			                .rolling(window=7, min_periods=1)
			                .mean()
			                .reset_index(level=0, drop=True)})
		            )

		self.full_df = self.full_df.merge(diesel_mean, how='left', on=['uuid', 'month', 'day'])
		self.full_df = self.full_df.merge(e5_mean, how='left', on=['uuid', 'month', 'day'])
		self.full_df = self.full_df.merge(e10_mean, how='left', on=['uuid', 'month', 'day'])

	def save_parquet(self, parquet_path=paths.PRICES_PARQUET_PATH):
		self.full_df.to_parquet(parquet_path, index=False)

	def full_processing(self, path=paths.PRICES_PARQUET_PATH, store_parquet=True):
		self.set_columns()
#		self.compute_means()
		if store_parquet:
			self.save_parquet(parquet_path=path)


class DataPipeline:
	def __init__(self, prices_processor: PricesProcessing):
		self.prices_processor = prices_processor

	def merge(self) -> pd.DataFrame:        ##This method will be used in the future for price predictions
		return self.prices_processor.full_df.merge(
				self.oil_processor.oil_df, how='left', on=["date", "month", "day"]
		)

	def process_all(self, store=True, final_path=paths.FINAL_PARQUET_PATH) -> pd.DataFrame:
		self.prices_processor.full_processing()
		final_df = self.prices_processor.full_df

		final_df.drop(columns=["id", "date"], inplace=True)
		final_df = final_df.sort_values(['uuid', 'month', 'day'])

		if store:
			os.makedirs(os.path.dirname(final_path), exist_ok=True)
			final_df.to_parquet(final_path, index=False)

		return final_df