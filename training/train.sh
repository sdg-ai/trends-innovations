#!/bin/bash
ANACONDA_ENV="tandic-training"
source activate $ANACONDA_ENV
echo "Running script with Anaconda environment: $ANACONDA_ENV"

date=$(date '+%Y-%m-%d %H:%M:%S')
python training/train.py --dataset "openai_annotated_data" --d "${date}" --model_name "distilbert-base-uncased"
python training/train.py --dataset "openai_annotated_data" --d "${date}" --model_name "roberta-base"
python training/train.py --dataset "openai_annotated_data" --d "${date}" --model_name "albert-base-v2"