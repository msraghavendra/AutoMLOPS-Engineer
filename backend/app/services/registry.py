import logging
import os
import socket
from urllib.parse import urlparse
import mlflow
from mlflow.tracking import MlflowClient
from app.config import settings

logger = logging.getLogger(__name__)

class RegistryService:
    def __init__(self):
        self.enabled = False
        try:
            # 1. Quick socket check to verify tracking server is actually reachable
            parsed = urlparse(settings.MLFLOW_TRACKING_URI)
            host = parsed.hostname or "localhost"
            port = parsed.port or (80 if parsed.scheme == "http" else 443)
            # Quick 1-second timeout connection check
            with socket.create_connection((host, port), timeout=1.0):
                pass

            # Configure tracking URI
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            # Try to ping/list experiments to check connectivity
            mlflow.search_experiments()
            self.enabled = True
            logger.info("Successfully connected to MLflow tracking server.")
        except Exception as e:
            logger.warning(
                f"Could not connect to MLflow tracking server at {settings.MLFLOW_TRACKING_URI}. "
                "Running in fallback mode (models will be saved locally and database tracked, but MLflow tracking is bypassed)."
            )
            self.enabled = False

    def log_and_register_model(
        self,
        experiment_name: str,
        run_name: str,
        model_name: str,
        metrics: dict,
        params: dict,
        model_path: str
    ) -> str:
        """
        Logs parameters, metrics, and models to MLflow, then registers the model.
        Returns the MLflow run ID if successful, otherwise None.
        """
        if not self.enabled:
            logger.warning("MLflow is disabled. Skipping remote registration.")
            return "local_run_no_mlflow"

        try:
            # Set or create experiment
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                
                # Log hyperparameters
                for key, val in params.items():
                    mlflow.log_param(key, val)
                
                # Log metrics (skip dictionaries or lists, log only numeric metrics)
                for key, val in metrics.items():
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(key, val)

                # Log model artifact
                # We can log the model file directly as a generic artifact
                mlflow.log_artifact(model_path, artifact_path="model")
                
                # Register model in registry
                model_uri = f"runs:/{run_id}/model"
                try:
                    mlflow.register_model(model_uri, model_name)
                except Exception as reg_err:
                    logger.warning(f"Failed to register model in MLflow registry: {reg_err}")
                
                return run_id
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            return "local_run_error"

# Singleton registry service instance
registry_service = RegistryService()
