import argparse
from src.model_training import run_training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process-data', action='store_true')
    args = parser.parse_args()
    run_training(data_proc=args.process_data)

if __name__ == '__main__':
    main()