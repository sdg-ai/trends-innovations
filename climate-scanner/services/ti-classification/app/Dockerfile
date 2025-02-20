FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/model.py app/run.py ./

# Create directory structure for training data
RUN mkdir -p training/results/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "run.py", "--server.address", "0.0.0.0"]