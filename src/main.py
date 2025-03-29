import argparse
from src.model_training import run_training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process-train_data', action='store_true', help='Process raw train_data instead of reading from parquet.')
    args = parser.parse_args()
    run_training(data_proc=args.process_data)

if __name__ == '__main__':
    main()