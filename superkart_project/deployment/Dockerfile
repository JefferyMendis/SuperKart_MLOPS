# superkart_project/deployment/Dockerfile
# Production-friendly Dockerfile for SuperKart Streamlit deployment.

# Use a slim Python base image to keep the container lightweight
FROM python:3.9-slim

# Prevent Python from buffering stdout/stderr (useful for logs)
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security best practices
RUN adduser --disabled-password --gecos "" --uid 1000 appuser

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and artifacts into the container
# --chown ensures correct ownership for the non-root user
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Expose Streamlit port (documentation purpose)
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
