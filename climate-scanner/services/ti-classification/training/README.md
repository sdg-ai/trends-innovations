# Trends and Innovations Model Training

This service handles the training of transformer-based models for text classification. It supports multiple model architectures, training configurations, and includes features for experiment tracking and model evaluation.

## Features

- Support for multiple transformer architectures (DistilBERT, ALBERT, RoBERTa)
- Configurable training parameters via YAML
- Experiment tracking with Weights & Biases
- Multi-seed training support
- Early stopping and learning rate scheduling
- Comprehensive metrics and evaluation

## Setup

### Environment

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ti-classifier
```

### Configuration

1. Copy `example.env` to `.env` and set your Weights & Biases API key:
```bash
cp example.env .env
```

2. Configure training parameters in `train_configs.yml`. Example configuration:
```yaml
top-50-by-precision-distilbert:
  model_name: "distilbert-base-uncased"
  lr: 0.00005
  num_warmup_steps: 500
  wandb:
    job_type_modifier: ""
    group_name_modifier: ""
  skip: False
  num_seeds: 5
  min_samples_per_label: 1
  only_top_n_categories_by: ["precision", 50]
```

## Training

### Single Configuration

Run training using the provided script:
```bash
./train.sh
```

This will:
1. Create a tmux session
2. Activate the conda environment
3. Run training with current timestamp
4. Auto-close the session when complete

### Parameter Sweep

For hyperparameter optimization:
```bash
./train_sweep.sh
```

## Project Structure

- `train.py`: Main training script with model training loop
- `utils/`: Helper functions and utilities
  - `config.py`: Configuration classes and parsing
  - `metrics.py`: Custom metrics collection
  - `utils.py`: General utility functions
- `data/`: Dataset handling and preprocessing
- `results/`: Training outputs and checkpoints
- `train_configs.yml`: Training configurations
- `results.ipynb`: Analysis notebook for training results

## Output Structure

Training results are saved under `results/checkpoints/`:
```
results/checkpoints/
├── model_name/
│   ├── tokenizer files
│   ├── label_encoder.pkl
│   └── seed_X/
│       └── model files
```

## Metrics

The training process tracks:
- Loss (training and validation)
- Accuracy
- F1 Score
- Precision
- Recall

All metrics are logged to Weights & Biases for experiment tracking.