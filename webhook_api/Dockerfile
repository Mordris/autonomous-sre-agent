# AIDA/webhook_api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file and install dependencies first.
COPY requirements.txt .
# Add timeout for consistency.
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application code into the container.
COPY . .

# The command to run when the container starts.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]