import argparse
import os
import sys

# Add the project root to the Python path BEFORE importing anything from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model_training import run_training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process-data', action='store_true')
    args = parser.parse_args()
    run_training(data_proc=args.process_data)

if __name__ == '__main__':
    main()
