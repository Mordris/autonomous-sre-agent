# AIDA/feedback_ui/app.py
# Full, final, and complete file content
import os
import streamlit as st
import mlflow
import pandas as pd
import json
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="AIDA: Feedback Console",
    page_icon="🤖",
    layout="wide"
)

# --- MLflow Connection ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

MLFLOW_EXPERIMENT_NAME = "AIDA_Investigations"

# --- Helper Functions ---

@st.cache_data(ttl=60)
def get_experiment_id(experiment_name):
    """Gets the experiment ID for a given experiment name."""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        return None
    except Exception:
        return None

@st.cache_data(ttl=10)
def get_all_runs(_exp_id):
    """Fetches all runs from an experiment and returns them as a DataFrame."""
    if not _exp_id:
        return pd.DataFrame()
        
    runs = client.search_runs(experiment_ids=[_exp_id], order_by=["start_time DESC"])
    
    processed_runs = []
    for run in runs:
        processed_runs.append({
            "Run ID": run.info.run_id,
            "Incident ID": run.data.tags.get("incident_id", "N/A"),
            "Alert Name": run.data.tags.get("alert_name", "N/A"),
            "Status": run.data.tags.get("investigation_status", "N/A"),
            "Feedback": run.data.tags.get("feedback_status", "Pending"),
            "Created At": pd.to_datetime(run.info.start_time, unit="ms")
        })
    return pd.DataFrame(processed_runs)

@st.cache_data(ttl=30)
def load_run_artifacts(run_id):
    """
    Safely downloads and loads the required JSON artifacts for a given run.
    The @st.cache_data decorator ensures this function completes fully before the
    rest of the script continues, preventing race conditions.
    """
    try:
        local_path = client.download_artifacts(run_id, ".", "/tmp/")
        
        with open(os.path.join(local_path, "final_report.json"), "r") as f:
            report = json.load(f)
        with open(os.path.join(local_path, "alert_payload.json"), "r") as f:
            alert = json.load(f)
            
        return alert, report
    except Exception as e:
        st.error(f"Failed to load artifacts for run {run_id}.")
        st.exception(e)
        return None, None


def submit_feedback(run_id, feedback_status, feedback_text=None):
    """Submits feedback to a specific MLflow run."""
    try:
        client.set_tag(run_id, "feedback_status", feedback_status)
        if feedback_text:
            client.log_text(run_id, feedback_text, "human_feedback.txt")
        
        # Clear the caches to force a reload of the table data on the next action
        st.cache_data.clear()
        
        st.success(f"Feedback successfully updated to '{feedback_status}'!")
        time.sleep(2) # Give user a moment to see the success message
        st.rerun() # Rerun the script to refresh the entire UI state
    except Exception as e:
        st.error(f"Failed to submit feedback: {e}")


# --- Main Application ---
st.title("🤖 AIDA: Investigation Feedback Console")
st.write("Review completed investigations and provide feedback to help AIDA learn.")

experiment_id = get_experiment_id(MLFLOW_EXPERIMENT_NAME)
if not experiment_id:
    st.warning(f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found. Please ensure the agent has run at least once.")
else:
    runs_df = get_all_runs(experiment_id)

    if runs_df.empty:
        st.info("No investigations found yet. Please trigger an alert to start an investigation.")
    else:
        st.header("Completed Investigations")
        st.info("Click on a row in the table below to select an investigation and provide feedback.")

        selection = st.dataframe(
            runs_df,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        if selection.selection.rows:
            selected_row_index = selection.selection.rows[0]
            selected_run_id = runs_df.iloc[selected_row_index]["Run ID"]

            st.divider()
            st.header(f"Investigation Details for Run: {selected_run_id}")
            
            alert, report = load_run_artifacts(selected_run_id)

            if alert and report:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Initial Alert")
                    st.json(alert)
                with col2:
                    st.subheader("AIDA's Final Conclusion")
                    st.info(report.get("final_conclusion", "No conclusion found."))

                with st.expander("Show Full Agent Trajectory (Chain of Thought)"):
                    st.json(report.get("full_trajectory", "No trajectory found."))

                # --- Final, Robust Feedback Form ---
                st.divider()
                st.subheader("Submit Your Feedback")
                
                current_feedback = runs_df[runs_df["Run ID"] == selected_run_id]["Feedback"].values[0]
                if current_feedback != "Pending":
                     st.success(f"Current feedback status: **{current_feedback}**")
                
                fb_col1, fb_col2 = st.columns(2)

                with fb_col1:
                    if st.button(
                        "👍 Approve - Diagnosis is Correct",
                        use_container_width=True,
                        type="primary",
                        disabled=(current_feedback == "Approved") # Disable if already approved
                    ):
                        submit_feedback(selected_run_id, "Approved")
                
                with fb_col2:
                    with st.form("correction_form"):
                        existing_correction = ""
                        if current_feedback == "Corrected":
                            try:
                                fb_path = client.download_artifacts(selected_run_id, "human_feedback.txt", "/tmp/")
                                with open(fb_path, "r") as f:
                                    existing_correction = f.read()
                            except Exception:
                                # File might not exist for older runs, fail gracefully
                                existing_correction = ""
                        
                        corrected_cause = st.text_area(
                            "✍️ Correct - The real root cause was...",
                            value=existing_correction,
                            height=150,
                            placeholder="e.g., The readiness probe failed due to a memory leak, not a downstream timeout."
                        )
                        
                        # The form button is only enabled if the user is submitting a NEW correction,
                        # or if they have CHANGED the text of an existing correction.
                        submitted = st.form_submit_button(
                            "Submit Correction",
                            use_container_width=True,
                            type="primary",
                            disabled=(current_feedback == "Corrected" and corrected_cause == existing_correction)
                        )
                        
                        if submitted:
                            if not corrected_cause.strip():
                                st.error("Please provide the corrected root cause text before submitting.")
                            else:
                                submit_feedback(selected_run_id, "Corrected", corrected_cause)