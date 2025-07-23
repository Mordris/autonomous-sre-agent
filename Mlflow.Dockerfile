# AIDA/Mlflow.Dockerfile
FROM python:3.11-slim

# MODIFIED: Add psycopg2-binary to connect to PostgreSQL
RUN pip install --no-cache-dir --default-timeout=100 mlflow gunicorn psycopg2-binary

# This is where artifacts (files) will be stored.
RUN mkdir /mlruns

EXPOSE 5000

# MODIFIED: A more robust command that uses the environment variables we set.
# This allows the server to handle more connections by increasing workers.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "${BACKEND_STORE_URI}", "--artifacts-destination", "${ARTIFACT_ROOT}", "--workers", "4"]
