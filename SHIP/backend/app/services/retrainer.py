import os
import logging
import pandas as pd
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Dataset, TrainedModel, Deployment, PredictionLog
from app.services import automl
from app.services.registry import registry_service
from app.services.deployment_manager import deployment_manager
from app.services.audit_logger import log_event

logger = logging.getLogger(__name__)

def train_model_async_task(db: Session, dataset_id: int, model_id: int):
    """
    FastAPI BackgroundTask function to perform AutoML model training and evaluation asynchronously.
    """
    # 1. Fetch records
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    db_model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    
    if not db_dataset or not db_model:
        logger.error(f"Dataset ID {dataset_id} or Model ID {model_id} not found during async training.")
        return

    try:
        # Update model status to TRAINING
        db_model.status = "TRAINING"
        db_model.status_message = "Training in progress..."
        db.commit()

        log_event(
            db, 
            event_type="TRAINING", 
            message=f"Starting AutoML training for model version {db_model.version} on dataset: {db_dataset.name}.",
            severity="INFO"
        )

        # 2. Load dataset CSV
        if not os.path.exists(db_dataset.file_path):
            raise FileNotFoundError(f"Dataset file missing at {db_dataset.file_path}")
        
        df = pd.read_csv(db_dataset.file_path)

        # 3. Retrieve any prediction feedback data with ground truth
        # Find active deployment for this model's dataset if it exists
        active_deployment = db.query(Deployment)\
            .join(TrainedModel)\
            .filter(TrainedModel.dataset_id == dataset_id, Deployment.status == "active")\
            .first()

        if active_deployment:
            # Query prediction logs with ground truths
            feedback_logs = db.query(PredictionLog)\
                .filter(PredictionLog.deployment_id == active_deployment.id, PredictionLog.actual_value.isnot(None))\
                .all()
            
            if feedback_logs:
                logger.info(f"Retraining includes {len(feedback_logs)} prediction feedback logs.")
                feedback_data = []
                for log in feedback_logs:
                    row = dict(log.input_data)
                    row[db_dataset.target_column] = log.actual_value
                    feedback_data.append(row)
                df_feedback = pd.DataFrame(feedback_data)
                
                # Align columns and concatenate
                # Cast actual_values to match df types if needed
                df = pd.concat([df, df_feedback], ignore_index=True)

        # Define file save path
        model_filename = f"model_v{db_model.version}_{db_model.algorithm}.joblib"
        model_save_path = os.path.join(settings.MODEL_DIR, model_filename)

        # 4. Fit AutoML models and tune hyperparameters
        results, best_algo_name, best_metrics = automl.train_and_evaluate(
            df=df,
            target_col=db_dataset.target_column,
            problem_type=db_dataset.problem_type,
            features_metadata=db_dataset.features_metadata,
            model_save_path=model_save_path
        )

        # 5. Log details to MLflow Registry
        run_name = f"run_v{db_model.version}_{db_model.algorithm}"
        # Merge metrics and details
        log_params = {
            "dataset_name": db_dataset.name,
            "dataset_rows": len(df),
            "algorithm": db_model.algorithm,
            "problem_type": db_dataset.problem_type
        }
        
        mlflow_run_id = registry_service.log_and_register_model(
            experiment_name="Ship_It_ML",
            run_name=run_name,
            model_name=db_model.model_name,
            metrics=best_metrics,
            params=log_params,
            model_path=model_save_path
        )

        # 6. Update TrainedModel status
        db_model.status = "DONE"
        db_model.metrics = best_metrics
        db_model.file_path = model_save_path
        db_model.mlflow_run_id = mlflow_run_id
        db_model.status_message = "Training successfully completed."
        db.commit()

        log_event(
            db, 
            event_type="TRAINING", 
            message=f"Model version {db_model.version} trained successfully using {db_model.algorithm}. Best Metric Score: {best_metrics.get('f1_score') or best_metrics.get('r2_score'):.4f}",
            severity="INFO"
        )

        # 7. Evaluate for automatic redeployment/promotion
        # If there is currently an active deployment for this dataset, check if we should auto-deploy
        if active_deployment:
            active_model = db.query(TrainedModel).filter(TrainedModel.id == active_deployment.model_id).first()
            if active_model and active_model.metrics:
                metric_key = "f1_score" if db_dataset.problem_type == "classification" else "r2_score"
                active_metric = active_model.metrics.get(metric_key, -1.0)
                new_metric = best_metrics.get(metric_key, -1.0)

                logger.info(f"Comparing performance: Active Model Metric: {active_metric:.4f}, New Model Metric: {new_metric:.4f}")

                # If new model outperforms the current active model, deploy it
                if new_metric > active_metric:
                    log_event(
                        db,
                        event_type="RETRAINING",
                        message=f"Retrained model v{db_model.version} ({new_metric:.4f}) outperformed active model v{active_model.version} ({active_metric:.4f}). Auto-promoting...",
                        severity="INFO"
                    )

                    # Swap deployments
                    active_deployment.status = "inactive"
                    
                    new_deployment = Deployment(
                        model_id=db_model.id,
                        status="active",
                        drift_threshold=active_deployment.drift_threshold,
                        performance_threshold=active_deployment.performance_threshold
                    )
                    db.add(new_deployment)
                    db.commit()
                    db.refresh(new_deployment)

                    # Swap in deployment manager
                    deployment_manager.swap_active_model(new_deployment.id, db_model.id, db_model.file_path)
                    
                    log_event(
                        db,
                        event_type="DEPLOYMENT",
                        message=f"Model v{db_model.version} deployed dynamically as active model.",
                        severity="INFO"
                    )
                else:
                    log_event(
                        db,
                        event_type="RETRAINING",
                        message=f"Retrained model v{db_model.version} ({new_metric:.4f}) did not outperform active model v{active_model.version} ({active_metric:.4f}). Promotion skipped.",
                        severity="INFO"
                    )

    except Exception as e:
        logger.error(f"Error during async training task: {e}", exc_info=True)
        db_model.status = "FAILED"
        db_model.status_message = str(e)
        db.commit()
        
        log_event(
            db, 
            event_type="TRAINING", 
            message=f"AutoML training failed for model version {db_model.version}: {str(e)}",
            severity="ERROR"
        )
