#!/bin/bash
ANACONDA_ENV="tandic-training"
source activate $ANACONDA_ENV
echo "Running script with Anaconda environment: $ANACONDA_ENV"

date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --d "${date}" --config_name baseline-distilbert-base-uncased-sweep