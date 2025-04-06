#data_fetch.py
from datetime import datetime, timedelta
import yfinance as yf
import os
import requests
import gzip
import shutil
import numpy as np
from tqdm import tqdm
from src.utils import paths

class BrentFetcher:
    def __init__(self, ticker="BZ=F", period_days=60):
        self.ticker = ticker
        self.period_days = period_days
        self.output_path = paths.OIL_CSV_PATH


    def fetch_brent(self):
        end_date=datetime.today().date()
        start_date = end_date - timedelta(days=self.period_days)
        data = yf.download(
            self.ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True
        )
        data.reset_index(inplace=True)
        data = data.iloc[1:].copy()
        data.columns = data.columns.get_level_values(0)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        data.to_csv(self.output_path, index=False)
        return data

class StationFetcher:
    def __init__(self, dump_url=paths.TANKERKOENIG_DUMP_URL, gz_filename = paths.DUMP_GZ_FILENAME,
                 output_filename = paths.DUMP_OUTPUT_FILENAME, get_dump = True, encoding='utf-8'):
        self.gz_filename = gz_filename
        self.output_filename = output_filename
        self.dump_url = dump_url
        if get_dump:
            with requests.get(dump_url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading dump")

                with open(gz_filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                progress_bar.close()
                if total_size != 0 and progress_bar.n != total_size:
                    print("WARNING: Download size mismatch!")

            file_size = os.path.getsize(gz_filename)
            with gzip.open(gz_filename, "rb") as f_in, open(output_filename, "wb") as f_out:
                with tqdm(total=file_size, unit='iB', unit_scale=True, desc="Writing dump") as pbar:
                    while True:
                        chunk = f_in.read(1024)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        pbar.update(len(chunk))
        with open(output_filename, "r", encoding=encoding) as f_in:
            total_lines = sum(1 for _ in open(output_filename, 'r', encoding=encoding))
            self.lines = []
            for line in tqdm(f_in, total=total_lines, desc="Reading dump"):
                self.lines.append(line)

    def create_prices_csv(self, encoding='utf-8', separator='\t'):
        start = None
        end = None
        table_found = False
        columns = [
            'id', 'uuid', 'e5', 'e10', 'diesel', 'date', 'change'
        ]

        for i, line in enumerate(self.lines):
            if line.startswith("COPY public.gas_station_information_history"):
                start = i + 1
                table_found = True
            elif (line.strip() == '\\.') and (table_found == True):
                end = i
                break

        with open(paths.STATION_PRICES_CSV, "w", encoding=encoding) as out_file:
            out_file.write(separator.join(columns) + '\n')
            for line in tqdm(self.lines[start:end], total=end-start, desc="Writing prices csv", unit="line"):
                out_file.write(line)

    def create_stations_csv(self, encoding='utf-8', separator='\t'):
        start = None
        end = None

        columns = [
            'id', 'version', 'version_time', 'name', 'brand', 'street', 'house_number', 'post_code', 'place',
            'public_holiday_identifier', 'lat', 'lng', 'price_in_import', 'price_changed', 'open_ts', 'ot_json',
            'station_in_import', 'first_active'
        ]

        for i, line in enumerate(self.lines):
            if line.startswith("COPY public.gas_station"):
                start = i + 1
            elif line.strip() == '\\.':
                end = i
                break

        with open(paths.STATION_DATA_CSV, "w", encoding=encoding) as out_file:
            out_file.write(separator.join(columns) + '\n')
            for line in tqdm(self.lines[start:end], total=end-start, desc="Writing stations_data.csv", unit="line"):
                out_file.write(line)

    def fetch_all(self, delete_dumps=True):
        self.create_prices_csv()
        self.create_stations_csv()
        if delete_dumps:
            for file in [paths.DUMP_OUTPUT_FILENAME, paths.DUMP_GZ_FILENAME]:
                try:
                    os.remove(file)
                    print(f"Deleted {file}")
                except FileNotFoundError:
                    print(f"Failed to delete {file}, skipping...")


def main():
    sf = StationFetcher()
    sf.fetch_all()
if __name__ == '__main__':
    main()
