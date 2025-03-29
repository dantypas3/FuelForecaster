import os.path
import unittest
import pandas as pd
from src import data_processing as dp
from src.utils import paths

#Test Paths
TEST_PARQUET_PATH = paths.test_data_path('parquets', 'test.parquet')
TEST_OIL_PARQUET = paths.test_data_path('parquets', 'test_oil.parquet')
TEST_PRICES_PARQUET = paths.test_data_path('parquets', 'test_prices.parquet')
TEST_FINAL_PARQUET = paths.test_data_path('parquets', 'test_final.parquet')

class TestOilProcessing(unittest.TestCase):

    def test_constructor_valid_csv(self):
        oil = dp.OilProcessing()
        self.assertIsNotNone(oil.oil_df)
        self.assertFalse(oil.oil_df.empty)

    def test_set_columns(self):
        test_oil = dp.OilProcessing()
        test_oil.set_columns()
        expected_cols = {"oil_price", "month", "day", "year"}
        actual_cols = set(test_oil.oil_df.columns)
        self.assertEqual(actual_cols, expected_cols, f"Unexpected columns: {actual_cols - expected_cols}")

    def test_add_rolling_means(self):
        """Test rolling mean calculations for 3d, 7d, and 14d averages."""
        test_oil = dp.OilProcessing()
        test_oil.set_columns()
        test_oil.add_rolling_means()
        self.assertIn("oil_3d_mean", test_oil.oil_df.columns)
        self.assertIn("oil_7d_mean", test_oil.oil_df.columns)
        self.assertIn("oil_14d_mean", test_oil.oil_df.columns)

    def test_save_parquet(self):
        """Test saving dataframe to a Parquet file."""
        test_oil = dp.OilProcessing()
        test_oil.save_parquet(TEST_OIL_PARQUET)
        self.assertTrue(os.path.exists(TEST_OIL_PARQUET))
        self.addCleanup(os.remove, TEST_OIL_PARQUET)  # Ensure cleanup

class TestPricesProcessing(unittest.TestCase):

    def test_constructor_valid(self):
        """Test that StationProcessing initializes correctly with valid data."""
        processor = dp.PricesProcessing()
        self.assertIsInstance(processor.full_df, pd.DataFrame)
        self.assertFalse(processor.full_df.empty)

    def test_datetime_columns(self):
        """Test that datetime-related columns were added."""
        processor = dp.PricesProcessing()
        processor.set_columns()
        for col in ['year', 'month', 'day', 'weekday', 'hour']:
            self.assertIn(col, processor.full_df.columns)

    def test_df_cleaning(self):
        """Test that 'date' is removed and filters applied."""
        processor = dp.PricesProcessing()
        processor.set_columns()
        processor.df_cleaning()
        self.assertNotIn("date", processor.full_df.columns)
        self.assertIn("station_id_encoded", processor.full_df.columns)
        self.assertTrue((processor.full_df['diesel'] >= 0.5).all())
        self.assertTrue((processor.full_df['diesel'] <= 3).all())

    def test_set_datetime_sin(self):
        """Test that sine features are added."""
        processor = dp.PricesProcessing()
        processor.set_columns()
        processor.df_cleaning()
        processor.set_datetime_sin()
        self.assertIn("hour_sin", processor.full_df.columns)
        self.assertIn("weekday_sin", processor.full_df.columns)

    def test_set_datetime_cos(self):
        """Test cosine features if explicitly called."""
        processor = dp.PricesProcessing()
        processor.set_columns()
        processor.df_cleaning()
        processor.set_datetime_cos()
        self.assertIn("hour_cos", processor.full_df.columns)
        self.assertIn("weekday_cos", processor.full_df.columns)

    def test_save_parquet(self):
        """Test saving dataframe to a Parquet file."""
        processor = dp.PricesProcessing()
        processor.save_parquet(TEST_PRICES_PARQUET)
        self.assertTrue(os.path.exists(TEST_PRICES_PARQUET))
        self.addCleanup(os.remove, TEST_PRICES_PARQUET)

class TestFunctions(unittest.TestCase):
    def test_final_process_csv_simple(self):
        """Test that processing a CSV files works as expected, without storing a parquet file."""
        oil = dp.OilProcessing()
        oil.full_processing(path=TEST_OIL_PARQUET)
        prices = dp.PricesProcessing()
        prices.full_processing(path=TEST_PRICES_PARQUET)
        final_df = dp.final_process_csv(oil, prices, path=TEST_FINAL_PARQUET)
        self.assertFalse(final_df.empty)
        for col in ['station_uuid', 'diesel', 'e5', 'e10', 'dieselchange', 'e5change',
       'e10change', 'year', 'month', 'day', 'weekday', 'hour',
       'station_id_encoded', 'hour_sin', 'weekday_sin', 'oil_price',
       'oil_7d_mean', 'oil_14d_mean']:
            self.assertIn(col, final_df.columns)
        self.addCleanup(os.remove, TEST_OIL_PARQUET)
        self.addCleanup(os.remove, TEST_PRICES_PARQUET)
        self.addCleanup(os.remove, TEST_FINAL_PARQUET)

    def test_final_process_csv_parquet(self):
        """Test that processing CSV files works as expected with storing a parquet file enabled."""
        oil = dp.OilProcessing()
        oil.full_processing()
        prices = dp.PricesProcessing()
        prices.full_processing()
        final_df = dp.final_process_csv(oil, prices, path=TEST_FINAL_PARQUET)
        self.assertFalse(final_df.empty)

        for col in ['station_uuid', 'diesel', 'e5', 'e10', 'dieselchange', 'e5change',
       'e10change', 'year', 'month', 'day', 'weekday', 'hour',
       'station_id_encoded', 'hour_sin', 'weekday_sin', 'oil_price',
       'oil_7d_mean', 'oil_14d_mean']:
            self.assertIn(col, final_df.columns)

        self.assertTrue(os.path.exists(TEST_FINAL_PARQUET))
        self.addCleanup(os.remove, TEST_FINAL_PARQUET)

    def test_final_process_parquets(self):
        """Test merging two parquet files without storing output."""
        oil = dp.OilProcessing()
        oil.full_processing(TEST_OIL_PARQUET)
        prices = dp.PricesProcessing()
        prices.full_processing(TEST_PRICES_PARQUET)
        final_df = dp.final_process_parquets(oil_parquet=TEST_OIL_PARQUET, prices_parquet=TEST_PRICES_PARQUET,
                                             path=TEST_FINAL_PARQUET)
        self.assertIsInstance(final_df, pd.DataFrame)
        self.assertFalse(final_df.empty)

        expected_columns = {
            'station_uuid', 'diesel', 'e5', 'e10', 'dieselchange', 'e5change',
            'e10change', 'year', 'month', 'day', 'weekday', 'hour',
            'station_id_encoded', 'hour_sin', 'weekday_sin', 'hour_cos', 'weekday_cos',
            'oil_price', 'oil_3d_mean', 'oil_7d_mean', 'oil_14d_mean'
        }

        actual_columns = set(final_df.columns)
        missing = expected_columns - actual_columns
        self.assertTrue(expected_columns.issubset(actual_columns), f"Missing columns: {missing}")
        self.addCleanup(os.remove, TEST_OIL_PARQUET)
        self.addCleanup(os.remove, TEST_PRICES_PARQUET)
        self.addCleanup(os.remove, TEST_FINAL_PARQUET)


if __name__ == "__main__":
    unittest.main()