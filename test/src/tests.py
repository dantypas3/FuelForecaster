import os.path
import unittest
import pandas as pd
import datetime as dt
from src import data_processing as dp

test_oil_csv = "../test_data/oil/oil.csv"
test_stations_csvs = "../test_data/stations/*/*"

class TestDataProcessing(unittest.TestCase):
    def test_constructor_no_arguments(self):                    #Assert that both dataframes are empty
        empty_dfs = dp.BaseDataProcessing()
        self.assertIs(empty_dfs.oil_df.empty, True)
        self.assertIs(empty_dfs.stations_df.empty, True)

    def test_constructor_only_oil_df(self):                     #Assert that oil_df is not empty
        oil_df = dp.BaseDataProcessing(csv_oil_path="../test_data/oil/oil.csv")
        self.assertIs(oil_df.oil_df.empty, False)
        self.assertIs(oil_df.stations_df.empty, True)

    def test_constructor_only_stations_df(self):                #Assert that stations_df is not empty
        stations_df = dp.BaseDataProcessing(csv_stations_path ="../test_data/stations/*/*")
        self.assertIs(stations_df.stations_df.empty, False)
        self.assertIs(stations_df.oil_df.empty, True)

    def test_constructor_all_arguments(self):                   #Assert that oil_df & stations_df is not empty
        full_df = dp.BaseDataProcessing("../test_data/oil/oil.csv", "../test_data/stations/*/*")
        self.assertIs(full_df.oil_df.empty, False)
        self.assertIs(full_df.stations_df.empty, False)

    def test_df_concat(self):                                   #Assert that the station csvs are succesfully
        test_stations_df = dp.BaseDataProcessing(csv_stations_path="../test_data/stations/*/*")
        df1 = pd.read_csv("../test_data/stations/1/2025-03-07-stations.csv")
        df2 = pd.read_csv("../test_data/stations/2/2025-03-08-stations.csv")
        df3 = pd.read_csv("../test_data/stations/3/2025-03-09-stations.csv")
        total_lines = df1.shape[0] + df2.shape[0] + df3.shape[0]
        self.assertEqual(test_stations_df.stations_df.shape[0], total_lines)

class TestOilProcessing(unittest.TestCase):
    def test_constructor_no_path(self):
        with self.assertRaises(TypeError):
            dp.OilProcessing()

    def test_constructor_no_csv(self):
        with self.assertRaises(FileNotFoundError):
            dp.OilProcessing('../test_data/oil/Empty')

    def test_constructor_no_directory(self):
        with self.assertRaises(FileNotFoundError):
            dp.OilProcessing('../test_data/oil/Wrong')

    def test_set_datetime(self):
        test_oil_df = dp.OilProcessing(test_oil_csv)
        test_oil_df.set_columns()
        self.assertTrue(dt.datetime, test_oil_df.oil_df["Day"])
        self.assertIn("month", test_oil_df.oil_df.columns)
        self.assertIn("day", test_oil_df.oil_df.columns)
        self.assertIn("year", test_oil_df.oil_df.columns)
        self.assertNotIn("Europe Brent Spot Price FOB  Dollars per Barrel", test_oil_df.oil_df.columns)
        self.assertIn("oil_price", test_oil_df.oil_df.columns)

    def test_set_columns_wrong_string(self):
        test_oil_df = dp.OilProcessing("../test_data/oil/oil_wrong.csv")
        with self.assertRaises(Exception):
            test_oil_df.set_columns()

    def test_df_cleaning(self):
        test_oil_df = dp.OilProcessing(test_oil_csv)
        test_oil_df.set_columns()
        test_oil_df.df_cleaning()
        self.assertNotIn("Day", test_oil_df.oil_df.columns)

    def test_add_7d_mean(self):
        test_oil_df = dp.OilProcessing(test_oil_csv)
        test_oil_df.set_columns()
        test_oil_df.add_7d_mean()
        self.assertIn("oil_7d_mean", test_oil_df.oil_df.columns)
        self.assertFalse(test_oil_df.oil_df["oil_7d_mean"].isna().all())

    def test_save_parquet(self):
        test_oil_df = dp.OilProcessing(test_oil_csv)
        test_oil_df.save_parquet("../../data/parquets/test_oil_df.parquet")
        self.assertTrue(os.path.exists('../../data/parquets/test_oil_df.parquet'))
        os.remove('../../data/parquets/test_oil_df.parquet')

##TODO stations tests