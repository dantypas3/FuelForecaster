import os.path
import unittest
import pandas as pd
import datetime as dt
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
        with self.assertRaises(KeyError):
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


if __name__ == "__main__":
    unittest.main()