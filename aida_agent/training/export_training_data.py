# AIDA/aida_agent/training/export_training_data.py
import os
import json
import mlflow
import logging
import re

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AIDA.DataExporter")

# --- Constants ---
MLFLOW_TRACKING_URI = "http://mlflow:5000"
INVESTIGATION_EXPERIMENT_NAME = "AIDA_Investigations"
OUTPUT_FILE = "aida_training_dataset.jsonl" # Using a clear, specific name

# --- MLflow Connection ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# --- Robust Parsing Logic ---
# This ensures our data is formatted perfectly before exporting
class SimpleToolAction:
    def __init__(self, tool, tool_input):
        self.tool = tool
        self.tool_input = tool_input

def parse_trajectory_step(step_str: str) -> tuple | None:
    tool_match = re.search(r"tool='([^']*)'", step_str)
    tool_input_match = re.search(r"tool_input=({.*?})", step_str, re.DOTALL)
    observation_split = re.split(r",\s+'", step_str, 1)
    if not tool_match or not tool_input_match or len(observation_split) < 2:
        return None
    try:
        tool_input = json.loads(observation_split[0].split("tool_input=")[1].replace("'", "\""))
    except:
        tool_input = observation_split[0].split("tool_input=")[1]
    return (SimpleToolAction(tool=tool_match.group(1), tool_input=tool_input), observation_split[1].strip(" '()"))

def format_prompt(alert: dict, trajectory: list) -> str:
    try:
        trajectory_str = "\n\n".join([f"Step {i+1}:\n- Tool: {step[0].tool}\n- Input: {step[0].tool_input}\n- Observation: {step[1]}" for i, step in enumerate(trajectory)])
    except Exception:
        trajectory_str = "Could not parse trajectory."
    return f"""<s><|user|>
You are an expert SRE agent. You were given the following alert:
**Alert:**
```json
{json.dumps(alert, indent=2)}```
You performed an investigation and took the following steps (your trajectory):
**Trajectory:**
{trajectory_str}
Based on all of this information, what is the final root cause analysis?<|end|>
<|assistant|>"""

def main():
    logger.info(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    try:
        exp = client.get_experiment_by_name(INVESTIGATION_EXPERIMENT_NAME)
    except Exception as e:
        logger.error(f"Could not connect to MLflow. Is the stack running? Run 'docker compose up -d'. Error: {e}")
        return

    all_runs = client.search_runs(experiment_ids=[exp.experiment_id])
    filtered_runs = [run for run in all_runs if run.data.tags.get("feedback_status") in ["Approved", "Corrected"]]
    
    if not filtered_runs:
        logger.warning("No runs with 'Approved' or 'Corrected' feedback found. No data to export.")
        return

    logger.info(f"Found {len(filtered_runs)} validated runs to export.")
    
    exported_samples = 0
    with open(OUTPUT_FILE, 'w') as f:
        for run in filtered_runs:
            run_id, feedback_status = run.info.run_id, run.data.tags.get("feedback_status")
            try:
                local_path = client.download_artifacts(run_id, ".", "/tmp/")
                with open(os.path.join(local_path, "final_report.json"), "r") as report_file:
                    report = json.load(report_file)
                with open(os.path.join(local_path, "alert_payload.json"), "r") as alert_file:
                    alert = json.load(alert_file)
                
                full_trajectory_str_list = report.get("full_trajectory", [])
                trajectory = [parse_trajectory_step(s) for s in full_trajectory_str_list if s]
                trajectory = [s for s in trajectory if s is not None]
                instruction = format_prompt(alert.get('raw_alert', alert), trajectory)
                
                response = ""
                if feedback_status == "Approved":
                    response = report.get("final_conclusion", "")
                elif feedback_status == "Corrected":
                    with open(os.path.join(local_path, "human_feedback.txt"), "r") as fb_file:
                        response = fb_file.read()

                if instruction and response:
                    training_sample = {"text": f"{instruction}{response}</s>"}
                    f.write(json.dumps(training_sample) + "\n")
                    exported_samples += 1
            except Exception:
                logger.error(f"Skipping run {run_id} due to processing error.", exc_info=True)
    
    logger.info(f"Successfully exported {exported_samples} samples to '{OUTPUT_FILE}'. This file is ready to be uploaded to Colab.")

if __name__ == "__main__":
    main()