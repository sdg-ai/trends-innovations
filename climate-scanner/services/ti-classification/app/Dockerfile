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
COPY app/model.py app/run.py app/api.py app/start.sh ./

# Make start script executable
RUN chmod +x start.sh

# Create directory structure for training data
RUN mkdir -p training/results/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose both Streamlit and FastAPI ports
EXPOSE 8501 8000

# Command to run both services
CMD ["./start.sh"]