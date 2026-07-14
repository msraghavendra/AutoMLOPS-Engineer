import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import TrainedModel, Deployment, PredictionLog, Dataset
from app.schemas import DeploymentRead, PredictionRequest, PredictionFeedback
from app.services.deployment_manager import deployment_manager
from app.services.audit_logger import log_event
from app.services.retrainer import train_model_async_task
from app.middleware import verify_api_key

router = APIRouter(prefix="/api/deployments", tags=["Deployments"])
logger = logging.getLogger(__name__)

@router.post("/deploy/{model_id}", response_model=DeploymentRead)
def deploy_model(model_id: int, db: Session = Depends(get_db)):
    """
    Deploy a specific trained model as the active prediction API.
    Updates database statuses and updates the in-memory active serving model.
    """
    # 1. Verify model exists and is trained successfully
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    if model.status != "DONE":
        raise HTTPException(status_code=400, detail="Cannot deploy a model that is not in 'DONE' state.")

    # 2. Deactivate all existing deployments
    db.query(Deployment).update({Deployment.status: "inactive"})
    db.commit()

    # 3. Check if deployment already exists for this model
    db_deployment = db.query(Deployment).filter(Deployment.model_id == model_id).first()
    if db_deployment:
        db_deployment.status = "active"
    else:
        db_deployment = Deployment(
            model_id=model_id,
            status="active",
            drift_threshold=0.3,
            performance_threshold=0.7
        )
        db.add(db_deployment)
    
    db.commit()
    db.refresh(db_deployment)

    # 4. Swap active model in DeploymentManager (Thread-safe)
    try:
        deployment_manager.swap_active_model(db_deployment.id, model.id, model.file_path)
    except Exception as e:
        logger.error(f"Error loading model to deployment manager: {e}")
        db_deployment.status = "inactive"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to load model file to serving memory: {str(e)}")

    # 5. Log deployment audit event
    log_event(
        db,
        event_type="DEPLOYMENT",
        message=f"Model v{model.version} ({model.algorithm}) deployed successfully. API is live.",
        severity="INFO"
    )

    return db_deployment


@router.get("/active", response_model=DeploymentRead)
def get_active_deployment(db: Session = Depends(get_db)):
    """
    Get current active deployment details.
    """
    deployment = db.query(Deployment).filter(Deployment.status == "active").first()
    if not deployment:
        raise HTTPException(status_code=404, detail="No active deployment found.")
    return deployment


@router.post("/predict")
def predict_endpoint(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Exposes live predictions using the active deployed model.
    Guarded by X-API-Key middleware security check.
    Logs inputs/outputs in PredictionLog for drift monitoring.
    """
    active_info = deployment_manager.get_active_model_info()
    if not active_info:
        raise HTTPException(status_code=503, detail="No active model deployed for predictions.")

    try:
        # Run prediction
        pred_output = deployment_manager.predict(request.inputs)
        
        # Log to database
        log_entry = PredictionLog(
            deployment_id=active_info["deployment_id"],
            input_data=request.inputs,
            prediction_output=pred_output
        )
        db.add(log_entry)

        # Increment prediction count
        db_deployment = db.query(Deployment).filter(Deployment.id == active_info["deployment_id"]).first()
        if db_deployment:
            db_deployment.prediction_count += 1

        db.commit()
        db.refresh(log_entry)

        return {
            "prediction_id": log_entry.id,
            "prediction": pred_output["prediction"],
            "raw_prediction": pred_output["raw_prediction"],
            "problem_type": pred_output["problem_type"]
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/feedback")
def log_prediction_feedback(
    feedback: PredictionFeedback,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Accepts actual ground truth labels for a prediction to monitor real-world accuracy.
    If accuracy falls below threshold, automatically schedules retraining.
    """
    # 1. Fetch prediction log
    pred_log = db.query(PredictionLog).filter(PredictionLog.id == feedback.prediction_id).first()
    if not pred_log:
        raise HTTPException(status_code=404, detail="Prediction log not found.")

    # 2. Update actual value
    pred_log.actual_value = str(feedback.actual_value)
    db.commit()

    # 3. Evaluate current accuracy
    # Find all predictions in this deployment with ground truth actual values
    deployment = db.query(Deployment).filter(Deployment.id == pred_log.deployment_id).first()
    if not deployment:
        return {"status": "success", "message": "Feedback logged."}

    labeled_logs = db.query(PredictionLog).filter(
        PredictionLog.deployment_id == deployment.id,
        PredictionLog.actual_value.isnot(None)
    ).all()

    # Calculate performance if we have enough logs (min 5)
    min_eval_logs = 5
    if len(labeled_logs) >= min_eval_logs:
        model = db.query(TrainedModel).filter(TrainedModel.id == deployment.model_id).first()
        dataset = db.query(Dataset).filter(Dataset.id == model.dataset_id).first()
        
        correct = 0
        total = len(labeled_logs)
        
        for log in labeled_logs:
            pred = str(log.prediction_output["prediction"])
            actual = str(log.actual_value)
            
            # Simple check for categorical/numeric match
            if dataset.problem_type == "classification":
                if pred.lower() == actual.lower():
                    correct += 1
            else:
                # For regression, calculate absolute error, or simulate accuracy by tolerance
                try:
                    p_val = float(pred)
                    a_val = float(actual)
                    # within 10% error threshold matches "correct"
                    if a_val != 0 and abs(p_val - a_val) / abs(a_val) < 0.15:
                        correct += 1
                    elif a_val == 0 and abs(p_val) < 0.15:
                        correct += 1
                except ValueError:
                    pass

        current_accuracy = float(correct / total)
        logger.info(f"Evaluated real-time model performance: {current_accuracy:.2f} (Threshold: {deployment.performance_threshold:.2f})")

        # Check performance drop
        if current_accuracy < deployment.performance_threshold:
            log_event(
                db,
                event_type="DRIFT",
                message=f"Model performance dropped to {current_accuracy:.2f} (threshold: {deployment.performance_threshold:.2f}). Triggering automatic retraining.",
                severity="WARNING"
            )

            # Launch Retraining in background
            new_model = TrainedModel(
                dataset_id=model.dataset_id,
                model_name=f"{dataset.name.replace('.csv', '')}_autoretrain_v{model.version + 1}",
                algorithm=model.algorithm,
                version=model.version + 1,
                status="PENDING",
                status_message="Performance drop auto-trigger retraining."
            )
            db.add(new_model)
            db.commit()
            db.refresh(new_model)

            background_tasks.add_task(train_model_async_task, db, dataset.id, new_model.id)
            
            return {
                "status": "warning",
                "message": f"Feedback logged. Model accuracy ({current_accuracy:.2f}) dropped below threshold ({deployment.performance_threshold:.2f}). Retraining queued.",
                "current_performance": current_accuracy
            }
            
        return {
            "status": "success",
            "message": "Feedback logged.",
            "current_performance": current_accuracy
        }

    return {"status": "success", "message": "Feedback logged."}
