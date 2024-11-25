# Use a slim Python 3.10 image as the base
FROM python:3.10.15-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock for dependency installation
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install pipenv and project dependencies
RUN pip install pipenv && pipenv install --system --deploy

# Copy application files to the container
COPY ["predict.py", "laptop_price_rf.bin", "./"]

# Expose the port the app runs on
EXPOSE 9696

# Set the entrypoint command to start the application using gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
