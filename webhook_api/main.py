# AIDA/webhook_api/main.py
import os
import redis
import json
import logging
import uuid
from fastapi import FastAPI, Request, HTTPException

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="AIDA Webhook Ingestion API",
    description="Receives alerts and dispatches them to the AIDA agent via a Redis queue.",
    version="1.0.0"
)

# --- Connect to Redis ---
# The hostname 'redis' is resolvable because all our services are in the same
# Docker Compose network, as defined in our docker-compose.yml file.
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_client = None
try:
    # decode_responses=True ensures that Redis returns strings, not bytes.
    redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
    # Check if the connection is alive.
    redis_client.ping()
    logger.info(f"Successfully connected to Redis at '{redis_host}'")
except redis.exceptions.ConnectionError as e:
    logger.error(f"FATAL: Could not connect to Redis: {e}. The API cannot queue jobs.")
    # In a real scenario, you might want the container to exit or have more complex retry logic.

# This is the name of the list in Redis that will act as our job queue.
AIDA_JOB_QUEUE = "aida_job_queue"

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"message": "AIDA Webhook API is running and ready to receive alerts."}

@app.post("/webhook", status_code=202, tags=["Incident Handling"])
async def receive_webhook(request: Request):
    """
    Receives alerts (e.g., from Prometheus Alertmanager), assigns an incident ID,
    and pushes a job to the Redis queue for the AIDA agent to process.
    """
    if not redis_client or not redis_client.ping():
        logger.error("Cannot process webhook: Redis service is unavailable.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Cannot connect to Redis queue.")

    try:
        alert_data = await request.json()
        logger.info("Webhook received.")

        # Create a job package for the AIDA agent.
        # We assign our own unique ID to track this incident internally.
        incident_id = str(uuid.uuid4())
        job_payload = {
            "incident_id": incident_id,
            "raw_alert": alert_data
        }

        # Push the job to the Redis list (which we use as a queue).
        # lpush adds the new job to the left (head) of the list.
        redis_client.lpush(AIDA_JOB_QUEUE, json.dumps(job_payload))
        logger.info(f"Dispatched job for incident_id '{incident_id}' to queue '{AIDA_JOB_QUEUE}'.")

        return {
            "status": "accepted",
            "incident_id": incident_id,
            "message": "AIDA has been dispatched to investigate."
        }

    except json.JSONDecodeError:
        logger.warning("Failed to process webhook: Invalid JSON payload.")
        raise HTTPException(status_code=400, detail="Bad Request: Invalid JSON payload.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")