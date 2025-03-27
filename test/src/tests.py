import os.path
import unittest
import pandas as pd
from src import data_processing as dp

TEST_OIL_CSV = "../test_data/oil/oil.csv"
TEST_OIL_WRONG_CSV = "../test_data/oil/oil_wrong.csv"
TEST_PARQUET_PATH = "../../data/parquets/test_oil_df.parquet"

class TestOilProcessing(unittest.TestCase):

    def test_constructor_no_path(self):
        """Test that initializing OilProcessing without a path raises ValueError."""
        with self.assertRaises(ValueError):
            dp.OilProcessing(None)

    def test_constructor_no_csv(self):
        """Test that initializing OilProcessing with a non-existent CSV raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            dp.OilProcessing('../test_data/oil/non_existent.csv')

    def test_constructor_valid_csv(self):
        """Test that OilProcessing initializes properly with a valid CSV."""
        self.assertTrue(os.path.exists(TEST_OIL_CSV), f"Test file not found: {TEST_OIL_CSV}")
        test_oil = dp.OilProcessing(TEST_OIL_CSV)
        self.assertFalse(test_oil.oil_df.empty)

    def test_set_columns(self):
        test_oil = dp.OilProcessing(TEST_OIL_CSV)
        # Do not call set_columns() again, since it's already been called in __init__
        self.assertIn("oil_price", test_oil.oil_df.columns)
        self.assertIn("month", test_oil.oil_df.columns)
        self.assertIn("day", test_oil.oil_df.columns)
        self.assertIn("year", test_oil.oil_df.columns)

    def test_set_columns_wrong_string(self):
        """Test if an exception is raised when the column name does not match the expected string."""
        with self.assertRaises(Exception):
            dp.OilProcessing(TEST_OIL_WRONG_CSV)

    def test_df_cleaning(self):
        """Test that the 'Day' column is removed correctly."""
        test_oil = dp.OilProcessing(TEST_OIL_CSV)
        self.assertNotIn("Day", test_oil.oil_df.columns)

    def test_add_rolling_means(self):
        """Test rolling mean calculations for 3d, 7d, and 14d averages."""
        test_oil = dp.OilProcessing(TEST_OIL_CSV)
        self.assertIn("oil_7d_mean", test_oil.oil_df.columns)
        self.assertIn("oil_14d_mean", test_oil.oil_df.columns)
        self.assertFalse(test_oil.oil_df["oil_7d_mean"].isna().all())

    def test_save_parquet(self):
        """Test saving dataframe to a Parquet file."""
        test_oil = dp.OilProcessing(TEST_OIL_CSV)
        test_oil.save_parquet(TEST_PARQUET_PATH)
        self.assertTrue(os.path.exists(TEST_PARQUET_PATH))
        self.addCleanup(os.remove, TEST_PARQUET_PATH)  # Ensure cleanup



TEST_STATION_DIR = "../test_data/stations"
TEST_FAKE_PATH = "../test_data/non_existent"
TEST_STATION_PARQUET_PATH = "../../data/parquets/test_station_df.parquet"

class TestStationProcessing(unittest.TestCase):
    class TestStationProcessing(unittest.TestCase):

        def test_constructor_no_path(self):
            """Test that initializing StationProcessing without a path raises ValueError."""
            with self.assertRaises(ValueError):
                dp.StationProcessing(None)

        def test_constructor_no_files(self):
            """Test that initializing StationProcessing with a non-existent path raises FileNotFoundError."""
            with self.assertRaises(FileNotFoundError):
                dp.StationProcessing(TEST_FAKE_PATH)

        def test_constructor_valid(self):
            """Test that StationProcessing initializes correctly with valid data."""
            processor = dp.StationProcessing(TEST_STATION_DIR)
            self.assertIsInstance(processor.full_df, pd.DataFrame)
            self.assertFalse(processor.full_df.empty)

        def test_datetime_columns(self):
            """Test that datetime-related columns were added."""
            processor = dp.StationProcessing(TEST_STATION_DIR)
            for col in ['year', 'month', 'day', 'weekday', 'hour']:
                self.assertIn(col, processor.full_df.columns)

        def test_df_cleaning(self):
            """Test that 'date' is removed and filters applied."""
            processor = dp.StationProcessing(TEST_STATION_DIR)
            self.assertNotIn("date", processor.full_df.columns)
            self.assertIn("station_id_encoded", processor.full_df.columns)
            self.assertTrue((processor.full_df['diesel'] >= 0.5).all())
            self.assertTrue((processor.full_df['diesel'] <= 3).all())

        def test_set_datetime_sin(self):
            """Test that sine features are added."""
            processor = dp.StationProcessing(TEST_STATION_DIR)
            self.assertIn("hour_sin", processor.full_df.columns)
            self.assertIn("weekday_sin", processor.full_df.columns)

        def test_set_datetime_cos(self):
            """Test cosine features if explicitly called."""
            processor = dp.StationProcessing(TEST_STATION_DIR)
            processor.set_datetime_cos()
            self.assertIn("hour_cos", processor.full_df.columns)
            self.assertIn("weekday_cos", processor.full_df.columns)

        def test_save_parquet(self):
            """Test saving processed station data to Parquet."""
            processor = dp.StationProcessing(TEST_STATION_DIR)
            processor.save_parquet()
            self.assertTrue(os.path.exists('../data/parquets//full_df.parquet'))
            # Optionally clean up if you want
            # self.addCleanup(os.remove, '../data/parquets//full_df.parquet')


if __name__ == "__main__":
    unittest.main()