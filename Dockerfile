# Base image with Python
FROM python:3.10-slim

# Install system dependencies (PortAudio and others)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run model training (optional: remove if model already trained)
RUN python model_training.py || true

# Expose the Streamlit port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
