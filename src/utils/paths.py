import os

from xyzservices.providers import data_path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def data_path(*args):
    return os.path.join(BASE_DIR, 'data', *args)

def test_data_path(*args):
    return os.path.join(BASE_DIR, 'test', 'test_data', *args)

def model_path(*args):
    return os.path.join(BASE_DIR, 'models', *args)

# Some common paths
OIL_CSV_PATH = data_path('train_data', 'oil', 'oil.csv')
STATION_PRICES_CSV = data_path('train_data', 'gas_stations', 'prices', 'station_prices.csv')
STATION_DATA_CSV = data_path('train_data', 'gas_stations', 'stations', 'stations_data.csv')
OIL_PARQUET_PATH = data_path( 'parquets', 'oil_df.parquet')
PRICES_PARQUET_PATH = data_path('parquets', 'prices_df.parquet')
FINAL_PARQUET_PATH = data_path('parquets', 'final_df.parquet')
MODEL_PATH = model_path('xgbr_trained.pkl')
TANKERKOENIG_DUMP_URL = "https://creativecommons.tankerkoenig.de/history/history.dump.gz"
DUMP_GZ_FILENAME = data_path('dump', 'history.dump.gz')
DUMP_OUTPUT_FILENAME = data_path('dump', 'history.dump')

