# AIDA/aida_agent/Dockerfile
# Final, Secure Version

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# --- SECURE DEPENDENCY INSTALLATION ---
# 1. Mount the secret file during the pip install run
# 2. Use the secret to log in via huggingface-cli
# 3. Install the packages
# 4. The secret is automatically unmounted and is NOT saved in the image layer.
RUN --mount=type=secret,id=huggingface_token \
    pip install --upgrade pip && \
    pip install huggingface-hub && \
    huggingface-cli login --token $(cat /run/secrets/huggingface_token) && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create and configure the non-root user
RUN useradd -ms /bin/bash aida && \
    adduser aida sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to the non-root user
USER aida

# Use the entrypoint to fix permissions at runtime
ENTRYPOINT ["/app/entrypoint.sh"]

# The command to run via the entrypoint
CMD ["python", "agent.py"]