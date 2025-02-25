# Trends and Innovations Classification Service

This service provides a complete pipeline for training and deploying text classification models for identifying trends and innovations in text data. The service is organized into several components, each handling different aspects of the classification process.

## Directory Structure

### `/annotation`
Text annotation service for creating and validating training data:
- Supports both OpenAI and Azure OpenAI models
- Provides tools for automated text annotation
- Includes majority voting and conflict resolution
- Benchmarking capabilities against human annotations

### `/app`
Web application and API for model inference:
- Streamlit-based web interface for interactive classification
- FastAPI-based REST API for programmatic access
- Support for multiple trained models and seeds
- Docker containerization for deployment

### `/training`
Model training infrastructure:
- Support for multiple transformer architectures (DistilBERT, ALBERT, RoBERTa)
- Configurable training parameters via YAML
- Experiment tracking with Weights & Biases
- Multi-seed training support
- Early stopping and learning rate scheduling

### `/dataset`
Contains the datasets used for training including the raw, human and ai annotated data.

### `/results`
Stores training outputs and model artifacts:
- Trained model checkpoints
- Evaluation metrics
- Training logs
- Model performance analysis

## Setup

The project includes a development environment configuration:
```bash
conda env create -f dev-environment.yml
```

## Component Documentation

Each component has its own detailed documentation:
- [Annotation Service Documentation](annotation/README.md)
- [Application Documentation](app/README.md)
- [Training Documentation](training/README.md)

## Workflow

1. **Data Annotation**: Use the annotation service to create labeled training data
2. **Model Training**: Train models using the training infrastructure
3. **Deployment**: Deploy trained models using the web app or API
4. **Inference**: Use the deployed service for text classification

## Development

For development work:
1. Set up the conda environment
2. Follow the component-specific documentation
3. Use Weights & Biases for experiment tracking
4. Follow the existing code structure and patterns