import os.path
import unittest
import pandas as pd
from src import data_processing as dp
from src.utils import paths

# Test Paths
TEST_PARQUET_PATH = paths.test_data_path('parquets', 'test.parquet')
TEST_OIL_PARQUET = paths.test_data_path('parquets', 'test_oil.parquet')
TEST_PRICES_PARQUET = paths.test_data_path('parquets', 'test_prices.parquet')
TEST_FINAL_PARQUET = paths.test_data_path('parquets', 'test_final.parquet')

class TestOilProcessing(unittest.TestCase):

    def setUp(self):
        self.oil = dp.OilProcessing()

    def test_constructor_valid_csv(self):
        self.assertIsNotNone(self.oil.oil_df)
        self.assertFalse(self.oil.oil_df.empty)

    def test_set_columns(self):
        self.oil.set_columns()
        expected_cols = {"oil_price", "month", "day", "year"}
        actual_cols = set(self.oil.oil_df.columns)
        self.assertEqual(actual_cols, expected_cols, f"Unexpected columns: {actual_cols - expected_cols}")

    def test_add_rolling_means(self):
        self.oil.set_columns()
        self.oil.add_rolling_means()
        self.assertIn("oil_3d_mean", self.oil.oil_df.columns)
        self.assertIn("oil_7d_mean", self.oil.oil_df.columns)
        self.assertIn("oil_14d_mean", self.oil.oil_df.columns)

    def test_save_parquet(self):
        self.oil.save_parquet(TEST_OIL_PARQUET)
        self.assertTrue(os.path.exists(TEST_OIL_PARQUET))
        self.addCleanup(os.remove, TEST_OIL_PARQUET)

class TestPricesProcessing(unittest.TestCase):

    def setUp(self):
        self.processor = dp.PricesProcessing()

    def test_constructor_valid(self):
        self.assertIsInstance(self.processor.full_df, pd.DataFrame)
        self.assertFalse(self.processor.full_df.empty)

    def test_datetime_columns(self):
        self.processor.set_columns()
        for col in ['year', 'month', 'day', 'weekday', 'hour']:
            self.assertIn(col, self.processor.full_df.columns)

    def test_df_cleaning(self):
        self.processor.set_columns()
        self.processor.df_cleaning()
        self.assertNotIn("date", self.processor.full_df.columns)
        self.assertIn("station_id_encoded", self.processor.full_df.columns)
        self.assertTrue((self.processor.full_df['diesel'] >= 0.5).all())
        self.assertTrue((self.processor.full_df['diesel'] <= 3).all())

    def test_set_datetime_sin(self):
        self.processor.set_columns()
        self.processor.df_cleaning()
        self.processor.set_datetime_sin()
        self.assertIn("hour_sin", self.processor.full_df.columns)
        self.assertIn("weekday_sin", self.processor.full_df.columns)

    def test_set_datetime_cos(self):
        self.processor.set_columns()
        self.processor.df_cleaning()
        self.processor.set_datetime_cos()
        self.assertIn("hour_cos", self.processor.full_df.columns)
        self.assertIn("weekday_cos", self.processor.full_df.columns)

    def test_save_parquet(self):
        self.processor.save_parquet(TEST_PRICES_PARQUET)
        self.assertTrue(os.path.exists(TEST_PRICES_PARQUET))
        self.addCleanup(os.remove, TEST_PRICES_PARQUET)

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.oil = dp.OilProcessing()
        self.prices = dp.PricesProcessing()
        self.pipeline = dp.DataPipeline(self.oil, self.prices)

    def test_pipeline_merge(self):
        self.oil.full_processing(path=TEST_OIL_PARQUET)
        self.prices.full_processing(path=TEST_PRICES_PARQUET)
        final_df = self.pipeline.merge()
        self.assertIsInstance(final_df, pd.DataFrame)
        self.assertFalse(final_df.empty)

    def test_pipeline_process_all(self):
        final_df = self.pipeline.process_all(store=True, final_path=TEST_FINAL_PARQUET)
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
        self.assertTrue(os.path.exists(TEST_FINAL_PARQUET))
        self.addCleanup(os.remove, TEST_FINAL_PARQUET)


if __name__ == "__main__":
    unittest.main()
