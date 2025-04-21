#!/bin/bash
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found under ~/miniconda3 or ~/anaconda3."
    echo "Please edit run_tune.sh to point at your install."
    exit 1
fi

echo "Activating environment fuel-forecaster"

conda activate fuel-forecaster

echo "Starting tuning"

export JOBLIB_TEMP_FOLDER="/home/dionyssis/PycharmProjects/FuelForecaster/tmp"

python src/main.py tune \
    --train-days 24 \
    --n-iter 25 \
    --n-jobs 20 \
    --cv-folds 2 \
    --random-state 1991 \
    --log-file "training_log.txt"

