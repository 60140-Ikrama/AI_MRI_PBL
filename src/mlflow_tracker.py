import os
import json
import time

def log_mlflow_run(run_name, params, metrics, artifacts=None):
    """
    Logs parameters, metrics, and artifact paths for a model/pipeline run.
    Integrates actual MLflow logging if installed and active; falls back to a local
    JSON database ('mlflow_local_runs.json') for standalone robustness.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Try actual MLflow logging
    mlflow_active = False
    try:
        import mlflow
        # Set local tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("PrognosAI-X_BrainTumor")
        
        with mlflow.start_run(run_name=run_name):
            # Log params
            mlflow.log_params(params)
            # Log metrics
            mlflow.log_metrics(metrics)
            # Log artifacts if any
            if artifacts:
                for art_name, art_path in artifacts.items():
                    if os.path.exists(art_path):
                        mlflow.log_artifact(art_path)
        mlflow_active = True
    except Exception as e:
        print(f"MLflow Warning: Could not log to server ({e}). Logging to local database instead.")
        
    # 2. Log to local JSON database (Fallback)
    local_db_path = "./mlflow_local_runs.json"
    
    run_entry = {
        "run_name": run_name,
        "timestamp": timestamp,
        "parameters": params,
        "metrics": metrics,
        "artifacts_logged": list(artifacts.keys()) if artifacts else [],
        "mlflow_integrated": mlflow_active
    }
    
    existing_runs = []
    if os.path.exists(local_db_path):
        try:
            with open(local_db_path, "r") as f:
                existing_runs = json.load(f)
        except Exception:
            existing_runs = []
            
    existing_runs.append(run_entry)
    
    try:
        with open(local_db_path, "w") as f:
            json.dump(existing_runs, f, indent=4)
    except Exception as e:
        print(f"Error saving local run database: {e}")
        
    return run_entry

def get_logged_runs():
    """
    Retrieves all logged runs from the local JSON database.
    """
    local_db_path = "./mlflow_local_runs.json"
    if os.path.exists(local_db_path):
        try:
            with open(local_db_path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []
