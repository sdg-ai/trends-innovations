# Article Annotation Service

This service provides automated article classification using LLMs (Large Language Models) for the Climate Scanner project. It processes articles by splitting them into sections and categorizing each section according to predefined categories.

## Features

- Automated article classification using OpenAI GPT models (supports both OpenAI and Azure OpenAI)
- Section-based analysis for more granular classification
- Configurable majority voting system for robust categorization
- Benchmarking capabilities against human annotations

## Setup

1. Create a `.env` file with your API credentials:
```
# OpenAI Configuration
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o  # optional, defaults to gpt-4o

# Azure OpenAI Configuration (if using Azure)
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=
```
If both OpenAI and Azure OpenAI are configured, the Azure settings will be used.

## Usage

Run the annotation service using the following command:

```bash
python run.py [options]
```

### Command Line Options

- `--data-dir`: Directory containing the dataset (default: ./dataset)
- `--num-category-groups`: Number of category groups (default: 1)
- `--num-majority-vote`: Number of votes required for majority (default: 1)
- `--conflict-resolution`: Strategy for resolving conflicts (choices: majority_vote, unsure; default: majority_vote)
- `--benchmark`: Enable benchmarking against human annotations

## Code Structure

- `run.py`: Main entry point and CLI interface
- `simple_annotation_runner.py`: Core annotation logic and LLM integration
- `utils.py`: Utility functions for data loading and preprocessing

### Key Components

#### Article Processing
- Articles are split into sections of configurable length (default: 3 sentences)
- Each section is processed independently for classification
- Results are aggregated using majority voting or custom conflict resolution