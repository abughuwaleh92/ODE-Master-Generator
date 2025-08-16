# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ python3-dev libgfortran5 build-essential \
    curl supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create directories
RUN mkdir -p /app/logs /app/static && chown -R root:root /app

# Expose ports
EXPOSE 8501 8000

# Add Supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start Supervisor (manages all processes)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
