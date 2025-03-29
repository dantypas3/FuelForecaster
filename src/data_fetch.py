#data_fetch.py
from datetime import datetime, timedelta
import yfinance as yf
import os
from src.utils import paths

class DataFetcher:
    def __init__(self, ticker="BZ=F", period_days=60, output_path=None):
        self.ticker = ticker
        self.period_days = period_days
        self.output_path = output_path or paths.OIL_CSV_PATH

    def fetch_brent(self):
        end_date=datetime.today().date()
        start_date = end_date - timedelta(days=self.period_days)
        data = yf.download(
            self.ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d")
        )
        data.reset_index(inplace=True)
        data = data.iloc[1:].copy()
        data.columns = data.columns.get_level_values(0)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        data.to_csv(self.output_path, index=False)
        return data