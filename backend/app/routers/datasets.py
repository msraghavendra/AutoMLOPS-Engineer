import os
import uuid
from typing import cast
import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.config import settings
from app.models import Dataset
from app.schemas import DatasetRead
from app.services.automl import inspect_dataset
from app.services.audit_logger import log_event
from app.services.ai_analyzer import analyze_dataset_with_ai, analyze_data_quality_with_ai

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
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Generate unique file path
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)

    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, engine='python')
        
        if df.empty:
            raise ValueError("Dataset is empty.")

        row_count = len(df)
        column_count = len(df.columns)

        # Call AI analyzer for profiling
        ai_result = analyze_dataset_with_ai(df, target_column)

        # Run Gemini data quality advisor
        quality_report = analyze_data_quality_with_ai(df)
        ai_result["quality_report"] = quality_report

        final_target_col = target_column or ai_result.get("suggested_target")
        problem_type = ai_result.get("suggested_problem_type", "classification")

        # Inspect dataset schema to get basic stats
        _, _, basic_meta = inspect_dataset(df, final_target_col)

        # Enrich basic metadata with AI roles and explanations
        column_analysis = ai_result.get("column_analysis", {})
        enriched_metadata = {}
        for col, meta in basic_meta.items():
            ai_col_info = column_analysis.get(col, {})
            enriched_metadata[col] = {
                "type": ai_col_info.get("type", meta["type"]),
                "null_count": meta["null_count"],
                "dtype": meta["dtype"],
                "stats": meta["stats"],
                "role": ai_col_info.get("role", "feature"),
                "explanation": ai_col_info.get("explanation", "Predictive feature.")
            }

        # Create database entry
        db_dataset = Dataset(
            name=file.filename,
            file_path=file_path,
            row_count=row_count,
            column_count=column_count,
            target_column=final_target_col,
            problem_type=problem_type,
            features_metadata=enriched_metadata,
            description=ai_result.get("description", ""),
            ai_analysis=ai_result
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


@router.get("/{dataset_id}/quality")
def get_dataset_quality(dataset_id: int, db: Session = Depends(get_db)):
    """
    Returns the Gemini-powered data quality report for a dataset.
    Includes detected issues, severity levels, recommended fixes, and an overall score.
    """
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    ai_analysis = db_dataset.ai_analysis or {}
    quality_report = ai_analysis.get("quality_report")

    if not quality_report:
        # Re-run quality analysis on the fly if it wasn't stored
        if not db_dataset.file_path or not os.path.exists(str(db_dataset.file_path)):
            raise HTTPException(status_code=404, detail="Dataset file not found on disk.")
        try:
            try:
                df = pd.read_csv(str(db_dataset.file_path))
            except Exception:
                df = pd.read_csv(str(db_dataset.file_path), engine='python')
            quality_report = analyze_data_quality_with_ai(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Quality analysis failed: {str(e)}")

    return quality_report


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

    if not os.path.exists(cast(str, db_dataset.file_path)):
        raise HTTPException(status_code=404, detail="Dataset file missing from disk.")

    try:
        try:
            df = pd.read_csv(cast(str, db_dataset.file_path), nrows=10)
        except Exception:
            df = pd.read_csv(cast(str, db_dataset.file_path), nrows=10, engine='python')
        # Handle inf/nan values to avoid JSON serialization error
        df_cleaned = df.fillna("")
        return {
            "columns": list(df.columns),
            "rows": df_cleaned.to_dict(orient="records"),
            "features_metadata": db_dataset.features_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading dataset preview: {str(e)}")
