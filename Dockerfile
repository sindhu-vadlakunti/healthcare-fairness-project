# Use a lightweight, stable Python 3.10 image suitable for ML
FROM python:3.10-slim

# Set environment variables for reproducibility
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by ML libraries (e.g., AIF360, LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only requirements to cache them in the Docker layer
# This optimizes build time if dependencies haven't changed
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir keeps the final image significantly smaller
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project codebase
COPY . .

# Ensure the results directory exists so volume mounting works seamlessly
RUN mkdir -p /app/results

# Specify the default command to run when the container starts
CMD ["python", "run.py"]
