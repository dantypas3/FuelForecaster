import unittest
import pandas as pd
from numpy.ma.core import ones_like

from src import data_processing as dp

class TestDataProcessing(unittest.TestCase):
    def test_constructor_no_arguments(self):                    #Assert that both dataframes are empty
        empty_dfs = dp.DataProcessing()
        self.assertIs(empty_dfs.oil_df.empty, True)
        self.assertIs(empty_dfs.stations_df.empty, True)

    def test_constructor_only_oil_df(self):
        oil_df = dp.DataProcessing(csv_oil_path="../test_data/oil/oil.csv")
        self.assertIs(oil_df.oil_df.empty, False)
        self.assertIs(oil_df.stations_df.empty, True)

    def test_constructor_only_stations_df(self):
        stations_df = dp.DataProcessing(csv_stations_path = "../test_data/stations/*/")
        self.assertIs(stations_df.stations_df.empty, False)
        self.assertIs(stations_df.oil_df.empty, True)
