#!/bin/bash
ANACONDA_ENV="tandic"
source activate $ANACONDA_ENV
echo "Running script with Anaconda environment: $ANACONDA_ENV"

date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --d "${date}"
