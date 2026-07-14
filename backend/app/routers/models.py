from typing import cast
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Form, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import pickle
import joblib
import tempfile
import uuid
import shutil
from app.config import settings
from app.database import get_db
from app.models import Dataset, TrainedModel
from app.schemas import TrainedModelRead
from app.services.retrainer import train_model_async_task, retrain_uploaded_model_async_task

router = APIRouter(prefix="/api/models", tags=["Models"])

@router.post("/train/{dataset_id}", response_model=TrainedModelRead)
def train_model(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    algorithm: str = Form("Random Forest"),  # Default algorithm, or AutoML will train all and pick this as base
    db: Session = Depends(get_db)
):
    """
    Triggers the AutoML training pipeline for a given dataset.
    Launches training in the background and returns a model job record.
    """
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Calculate model version
    existing_versions = db.query(TrainedModel).filter(TrainedModel.dataset_id == dataset_id).count()
    next_version = existing_versions + 1

    # Create model record in PENDING state
    db_model = TrainedModel(
        dataset_id=dataset_id,
        model_name=f"{dataset.name.replace('.csv', '')}_model_v{next_version}",
        algorithm=algorithm,
        version=next_version,
        status="PENDING",
        status_message="Training job is queued."
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    # Queue background task
    background_tasks.add_task(train_model_async_task, db, dataset_id, cast(int, db_model.id))

    return db_model


@router.get("/train/status/{model_id}", response_model=TrainedModelRead)
def get_training_status(model_id: int, db: Session = Depends(get_db)):
    """
    Checks the async training status of a specific model.
    """
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    return model


@router.get("", response_model=list[TrainedModelRead])
def list_models(db: Session = Depends(get_db)):
    """
    List all trained models and their status / metrics.
    """
    return db.query(TrainedModel).order_by(TrainedModel.created_at.desc()).all()


@router.post("/upload-custom", response_model=TrainedModelRead)
async def upload_custom_model(
    dataset_id: int = Form(...),
    model_name: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Upload an existing scikit-learn model (.joblib) and retrain it on a selected dataset.
    """
    if not file.filename or not file.filename.endswith(".joblib"):
        raise HTTPException(status_code=400, detail="Only .joblib model files are supported.")

    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Save uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save uploaded model: {str(e)}")

    # Calculate model version
    existing_versions = db.query(TrainedModel).filter(TrainedModel.dataset_id == dataset_id).count()
    next_version = existing_versions + 1

    # Create model record in PENDING state
    db_model = TrainedModel(
        dataset_id=dataset_id,
        model_name=model_name or f"{dataset.name.replace('.csv', '')}_custom_v{next_version}",
        algorithm="Custom Uploaded",
        version=next_version,
        status="PENDING",
        status_message="Custom model retrain job queued."
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    # Queue background task
    background_tasks.add_task(retrain_uploaded_model_async_task, db, dataset_id, cast(int, db_model.id), temp_path)

    return db_model


@router.get("/download/{model_id}")
def download_model(
    model_id: int,
    background_tasks: BackgroundTasks,
    format: str = "joblib",
    db: Session = Depends(get_db)
):
    """
    Download the trained model in either .joblib or .pkl format.
    """
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    
    if model.status != "DONE":
        raise HTTPException(status_code=400, detail=f"Model training status is {model.status}, not ready for download.")

    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(status_code=404, detail="Model file not found on disk.")

    if format.lower() == "pkl":
        try:
            model_data = joblib.load(model.file_path)
            
            # Create a temporary file to save the pickled model
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            temp_file_path = temp_file.name
            temp_file.close()
            
            with open(temp_file_path, "wb") as f:
                pickle.dump(model_data, f)
                
            # Clean up the temp file after the request is finished
            background_tasks.add_task(os.remove, temp_file_path)
            
            return FileResponse(
                path=temp_file_path,
                filename=f"{model.model_name}.pkl",
                media_type="application/octet-stream"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to convert model to pickle format: {str(e)}")
            
    elif format.lower() == "joblib":
        return FileResponse(
            path=model.file_path,
            filename=f"{model.model_name}.joblib",
            media_type="application/octet-stream"
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Supported formats are 'joblib' and 'pkl'.")
