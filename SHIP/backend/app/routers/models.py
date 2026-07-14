from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Dataset, TrainedModel
from app.schemas import TrainedModelRead
from app.services.retrainer import train_model_async_task

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
    background_tasks.add_task(train_model_async_task, db, dataset_id, db_model.id)

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
