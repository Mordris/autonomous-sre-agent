# AIDA Training Pipeline

This directory contains the scripts and notebooks for fine-tuning the AIDA agent model.

## Workflow

The training process is a two-step hybrid cloud/local workflow:

### 1. Data Export (Local)

The `export_data.py` script is run locally inside the running Docker environment. It connects to the project's MLflow database, finds all investigation runs that have been marked as "Approved" or "Corrected" via the Feedback UI, and exports them into a single, clean training file named `aida_training_dataset.jsonl`.

**To run:**

```bash
# Ensure the main AIDA stack is running
docker compose up -d

# Execute the export script
docker compose exec aida_agent python3 training/export_data.py
```
