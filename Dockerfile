# Use Google's Docker Hub mirror if Docker Hub is blocked or slow (connection refused)
# Fallback: use python:3.11-slim when Docker Hub is reachable
FROM mirror.gcr.io/library/python:3.11-slim

WORKDIR /app

# Install poppler-utils for pdf2image (PDF → images for OCR fallback)
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/ templates/
# uploads/ and data/ are created at runtime by app.py; use volumes in docker-compose to persist
RUN mkdir -p uploads data

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]





