# AIDA/aida_agent/agent.py
# FINAL VERSION - All fixes included

import os
import redis
import json
import logging
import mlflow
import torch
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from typing import Any, List, Optional

# Using the correct, modern wrapper for Hugging Face integration
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from tools import all_tools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AIDA.Agent")

# --- Environment & Constants ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
AIDA_JOB_QUEUE = "aida_job_queue"
MLFLOW_EXPERIMENT_NAME = "AIDA_Investigations"
BASE_MODEL_NAME = "google/gemma-2b-it"
ADAPTER_PATH = "./training/aida-gemma-2b-sre-adapter-v1"

# --- System Prompt for the ReAct Agent ---
SYSTEM_PROMPT = """
You are AIDA, the Autonomous Incident Diagnostic Agent, a Level 1 Site Reliability Engineer (SRE).
Your sole mission is to investigate and determine the root cause of production alerts.

**Your Persona:**
- You are technical, methodical, and precise.
- You communicate in clear, concise language.
- You form a hypothesis and use tools to gather evidence to prove or disprove it.

**RESPONSE FORMAT:**
You MUST respond in one of two formats.

**FORMAT 1: You need to use a tool**
Thought: Your reasoning for why you need to use a tool.
Action: The name of the single tool to use (from the provided list).
Action Input: The input for the tool.

**FORMAT 2: You have the final answer**
Thought: Your reasoning for why you have the final answer.
Final Answer: The detailed root cause of the incident.

You must choose one format. You cannot provide both an Action and a Final Answer in the same response.

**Rules of Engagement:**
1.  **Analyze the Alert:** Start by carefully reading the user's input, which will be a JSON object containing the production alert details.
2.  **Consult Runbooks First:** Your first step should ALWAYS be to use the `search_runbooks` tool.
3.  **Use Your Tools:** Follow the procedure from the runbook. Use the provided tools to gather metrics, logs, and system states.
4.  **Reason Step-by-Step:** Think through your findings at each step. If you get new information, state how it changes your hypothesis.
5.  **Synthesize and Conclude:** Once you have gathered sufficient evidence, switch to FORMAT 2 and provide the final answer.
"""

# --- Helper Functions (Unchanged) ---
def connect_to_redis():
    try:
        client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
        client.ping()
        logger.info(f"Successfully connected to Redis at '{REDIS_HOST}'")
        return client
    except redis.exceptions.ConnectionError as e:
        logger.error(f"FATAL: Could not connect to Redis: {e}. AIDA cannot start.")
        return None

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking server is set to: {MLFLOW_TRACKING_URI}")

def get_or_create_experiment(experiment_name: str) -> str:
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        return experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Could not get or create MLflow experiment '{experiment_name}'.", exc_info=True)
        raise

def process_incident(job_data: dict, agent_executor, experiment_id: str):
    incident_id = job_data.get("incident_id", "unknown-incident")
    alert_data = job_data.get("raw_alert", {})
    logger.info(f"[{incident_id}] Starting investigation...")

    input_prompt = f"""
{SYSTEM_PROMPT}

An alert has fired. Here is the alert data in JSON format:
{json.dumps(alert_data, indent=2)}

Please investigate and determine the root cause, following all rules. Begin!
"""

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Incident-{incident_id}") as run:
        run_id = run.info.run_id
        mlflow.set_tag("incident_id", incident_id)
        alert_name = alert_data.get('alerts', [{}])[0].get('labels', {}).get('alertname', 'Unknown Alert')
        mlflow.set_tag("alert_name", alert_name)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_payload_path = os.path.join(tmpdir, "alert_payload.json")
            with open(alert_payload_path, "w") as f:
                json.dump(alert_data, f, indent=4)
            mlflow.log_artifact(alert_payload_path)
            
            try:
                response = agent_executor.invoke({"input": input_prompt})
                logger.info(f"[{incident_id}] Agent finished investigation.")
                final_report = {
                    "final_conclusion": response.get('output', 'No output from agent.'),
                    "full_trajectory": [str(x) for x in response.get('intermediate_steps', [])]
                }
                mlflow.log_dict(final_report, "final_report.json")
                mlflow.set_tag("investigation_status", "complete_success")
            except Exception as e:
                logger.error(f"[{incident_id}] Agent investigation failed!", exc_info=True)
                mlflow.log_param("error_message", str(e))
                mlflow.set_tag("investigation_status", "complete_failure")
        logger.info(f"[{incident_id}] Investigation logged to MLflow run {run_id}.")


def main():
    """Main worker loop. Initializes the agent and listens for jobs."""
    redis_client = connect_to_redis()
    if not redis_client: return
    setup_mlflow()
    experiment_id = get_or_create_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("--- Initializing Self-Hosted AIDA Agent (Gemma-2B) ---")
    
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=quant_config, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    if not os.path.isdir(ADAPTER_PATH):
        logger.critical(f"FATAL: Adapter path not found at '{ADAPTER_PATH}'.")
        return
        
    logger.info(f"Loading and merging LoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False # The critical fix
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, all_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True # Crucial for robustness
    )

    logger.info("AIDA Agent (Self-Hosted with Gemma-2B) is fully initialized. Listening on queue...")
    while True:
        try:
            _, job_json = redis_client.brpop(AIDA_JOB_QUEUE)
            job_data = json.loads(job_json)
            process_incident(job_data, agent_executor, experiment_id)
        except Exception:
            logger.error("A critical error occurred in the main loop.", exc_info=True)

if __name__ == "__main__":
    main()