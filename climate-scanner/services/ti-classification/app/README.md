# Trends and Innovations Classification App

This service provides both a web interface (Streamlit) and an API (FastAPI) for classifying text using trained DistilBERT models. The service is containerized and can be easily deployed using Docker.

## Features

- Web interface for interactive text classification
- RESTful API for programmatic access
- Support for multiple trained models and seeds
- Real-time classification with probability scores
- Docker containerization

## Setup

### Prerequisites

- Docker
- Python 3.11 or later (if running locally)
- Trained model checkpoints in the correct directory structure

### Directory Structure

```
training/results/checkpoints/
├── model_name_1/
│   ├── tokenizer files
│   ├── label_encoder.pkl
│   └── seed_1/
│       └── model files
└── model_name_2/
    ├── tokenizer files
    ├── label_encoder.pkl
    └── seed_2/
        └── model files
```

### Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your trained model checkpoints in the `training/results/checkpoints` directory

3. Start both services:
```bash
./start.sh
```

### Using Docker

1. Build the Docker image:
```bash
docker build -t ti-classifier .
```

2. Run the container:
```bash
docker run -p 8501:8501 -p 8000:8000 -v /path/to/checkpoints:/app/training/results/checkpoints ti-classifier
```

## Usage

### Web Interface

Access the Streamlit interface at `http://localhost:8501`:
1. Select a model and seed from the dropdown menus
2. Enter text in the input area
3. View classification results with probability scores

### API Endpoints

The FastAPI service runs on `http://localhost:8000`:

- `GET /models`: List available models and their seeds
  ```json
  {
    "available_models": {
      "model_name": ["seed_1", "seed_2"]
    },
    "current_model": {
      "model_name": "current_model",
      "seed": "current_seed"
    }
  }
  ```

- `POST /set_model`: Set the active model
  ```json
  {
    "model_name": "model_name",
    "seed": "seed_1"
  }
  ```

- `POST /predict`: Classify text
  ```json
  {
    "text": "Your text to classify"
  }
  ```

## Components

- `run.py`: Streamlit web interface
- `api.py`: FastAPI REST API
- `model.py`: DistilBERT model wrapper
- `start.sh`: Script to run both services
- `Dockerfile`: Container configuration