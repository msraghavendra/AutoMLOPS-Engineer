import os
import uuid
import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.config import settings
from app.models import Dataset
from app.schemas import DatasetRead
from app.services.automl import inspect_dataset
from app.services.audit_logger import log_event

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

@router.post("/upload", response_model=DatasetRead)
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a CSV dataset.
    Saves the file, performs automated data profiling, determines target column
    and problem type (classification vs regression), and stores metadata.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Generate unique file path
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)

    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Load file with pandas
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("Dataset is empty.")

        row_count = len(df)
        column_count = len(df.columns)

        # Inspect dataset schema and deduce task type
        final_target_col, problem_type, features_metadata = inspect_dataset(df, target_column)

        # Create database entry
        db_dataset = Dataset(
            name=file.filename,
            file_path=file_path,
            row_count=row_count,
            column_count=column_count,
            target_column=final_target_col,
            problem_type=problem_type,
            features_metadata=features_metadata
        )
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)

        # Log audit log
        log_event(
            db, 
            event_type="SYSTEM", 
            message=f"Dataset '{file.filename}' uploaded successfully. Target: {final_target_col}, Task: {problem_type}, Rows: {row_count}",
            severity="INFO"
        )

        return db_dataset

    except Exception as e:
        # Cleanup file if saved
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")


@router.get("", response_model=list[DatasetRead])
def list_datasets(db: Session = Depends(get_db)):
    """
    List all uploaded datasets.
    """
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


@router.get("/{dataset_id}/preview")
def get_dataset_preview(dataset_id: int, db: Session = Depends(get_db)):
    """
    Fetches the first 10 rows of the dataset as a preview.
    """
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not os.path.exists(db_dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file missing from disk.")

    try:
        df = pd.read_csv(db_dataset.file_path, nrows=10)
        # Handle inf/nan values to avoid JSON serialization error
        df_cleaned = df.fillna("")
        return {
            "columns": list(df.columns),
            "rows": df_cleaned.to_dict(orient="records"),
            "features_metadata": db_dataset.features_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading dataset preview: {str(e)}")
