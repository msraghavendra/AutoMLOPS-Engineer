from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

class DatasetBase(BaseModel):
    name: str

class DatasetCreate(DatasetBase):
    file_path: str
    row_count: int
    column_count: int
    target_column: str
    problem_type: str
    features_metadata: Dict[str, Any]

class DatasetRead(DatasetBase):
    id: int
    file_path: str
    row_count: int
    column_count: int
    target_column: str
    problem_type: str
    features_metadata: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class TrainedModelBase(BaseModel):
    model_name: str
    algorithm: str

class TrainedModelRead(TrainedModelBase):
    id: int
    dataset_id: int
    metrics: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    version: int
    mlflow_run_id: Optional[str] = None
    created_at: datetime
    status: str
    status_message: Optional[str] = None

    class Config:
        from_attributes = True


class DeploymentRead(BaseModel):
    id: int
    model_id: int
    status: str
    deployed_at: datetime
    prediction_count: int
    drift_threshold: float
    performance_threshold: float
    model: Optional[TrainedModelRead] = None

    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    inputs: Dict[str, Any]


class PredictionFeedback(BaseModel):
    prediction_id: int
    actual_value: Any


class DriftReportRead(BaseModel):
    id: int
    deployment_id: int
    drift_score: float
    metrics: Dict[str, Any]
    checked_at: datetime
    has_drift: bool

    class Config:
        from_attributes = True


class AuditLogRead(BaseModel):
    id: int
    event_type: str
    message: str
    severity: str
    created_at: datetime

    class Config:
        from_attributes = True


class RetrainSettings(BaseModel):
    drift_threshold: Optional[float] = None
    performance_threshold: Optional[float] = None
