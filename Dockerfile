# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies using Poetry
RUN poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY app/ ./app/
COPY input_images/ ./input_images/
COPY image_metadata.json ./
COPY image_list.json ./

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run Streamlit app
CMD ["poetry", "run" ,"streamlit", "run", "app/__main__.py"]

