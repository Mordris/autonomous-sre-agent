# AIDA/feedback_ui/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the application code
COPY . .

# The command to run the Streamlit application.
# It will run the file named 'app.py'.
# The --server.port and --server.address flags are critical for making it
# accessible from outside the Docker container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]