import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Deployment, TrainedModel, Dataset, PredictionLog, DriftReport
from app.schemas import DriftReportRead, RetrainSettings
from app.services.drift import calculate_data_drift
from app.services.audit_logger import log_event
from app.services.retrainer import train_model_async_task

router = APIRouter(prefix="/api/monitoring", tags=["Monitoring & Drift"])
logger = logging.getLogger(__name__)

@router.get("/drift")
def check_drift_endpoint(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Computes data drift on the active model.
    Compares baseline training features against production prediction logs.
    If drift ratio exceeds threshold, triggers automatic retraining and redeployment.
    """
    # 1. Fetch active deployment
    deployment = db.query(Deployment).filter(Deployment.status == "active").first()
    if not deployment:
        raise HTTPException(status_code=404, detail="No active model deployment found.")

    model = db.query(TrainedModel).filter(TrainedModel.id == deployment.model_id).first()
    dataset = db.query(Dataset).filter(Dataset.id == model.dataset_id).first()

    # 2. Load reference dataset
    try:
        ref_df = pd.read_csv(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load reference dataset: {str(e)}")

    # 3. Load production prediction logs
    pred_logs = db.query(PredictionLog).filter(PredictionLog.deployment_id == deployment.id).all()
    if not pred_logs:
        return {
            "drift_score": 0.0,
            "metrics": {},
            "has_drift": False,
            "message": "No production predictions logged yet. Run predictions first to calculate drift."
        }

    # Extract input data into a DataFrame
    current_data = [log.input_data for log in pred_logs]
    current_df = pd.DataFrame(current_data)

    # 4. Calculate drift using Evidently AI
    drift_score, feature_metrics, has_drift, message = calculate_data_drift(
        reference_df=ref_df,
        current_df=current_df,
        target_column=dataset.target_column,
        drift_threshold=deployment.drift_threshold
    )

    # 5. Handle drift detection actions
    # Save report to DB if we successfully calculated (logs >= 5) and check is active
    if feature_metrics:
        # Check if we already logged a report for this check cycle to avoid duplicates
        # In a real app we might throttle this. Here we save to track history.
        db_report = DriftReport(
            deployment_id=deployment.id,
            drift_score=drift_score,
            metrics=feature_metrics,
            has_drift=has_drift
        )
        db.add(db_report)
        db.commit()

        if has_drift:
            # Log warning event
            log_event(
                db,
                event_type="DRIFT",
                message=f"Data drift detected! Drift ratio is {drift_score:.2f} (threshold: {deployment.drift_threshold:.2f}). Triggering auto-retraining.",
                severity="WARNING"
            )

            # Trigger automatic retraining as background task
            # Calculate new model version
            next_version = db.query(TrainedModel).filter(TrainedModel.dataset_id == dataset.id).count() + 1
            new_model = TrainedModel(
                dataset_id=dataset.id,
                model_name=f"{dataset.name.replace('.csv', '')}_autotune_v{next_version}",
                algorithm=model.algorithm,
                version=next_version,
                status="PENDING",
                status_message="Auto-retrained due to data drift."
            )
            db.add(new_model)
            db.commit()
            db.refresh(new_model)

            background_tasks.add_task(train_model_async_task, db, dataset.id, new_model.id)
            
            message += " Data drift exceeded threshold. Retraining job has been queued."

    return {
        "drift_score": drift_score,
        "metrics": feature_metrics,
        "has_drift": has_drift,
        "message": message
    }


@router.post("/thresholds")
def configure_thresholds(settings: RetrainSettings, db: Session = Depends(get_db)):
    """
    Update drift and performance threshold values for the active deployment.
    """
    deployment = db.query(Deployment).filter(Deployment.status == "active").first()
    if not deployment:
        raise HTTPException(status_code=404, detail="No active deployment found to configure.")

    if settings.drift_threshold is not None:
        deployment.drift_threshold = settings.drift_threshold
    if settings.performance_threshold is not None:
        deployment.performance_threshold = settings.performance_threshold

    db.commit()
    db.refresh(deployment)

    log_event(
        db,
        event_type="SYSTEM",
        message=f"Deployment thresholds updated: Drift = {deployment.drift_threshold}, Performance = {deployment.performance_threshold}",
        severity="INFO"
    )

    return {
        "status": "success",
        "drift_threshold": deployment.drift_threshold,
        "performance_threshold": deployment.performance_threshold
    }
