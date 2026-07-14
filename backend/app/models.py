import datetime
from typing import Any, Dict, Optional
from sqlalchemy import Integer, String, Float, DateTime, ForeignKey, Boolean, JSON, Text
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    file_path: Mapped[str] = mapped_column(String)
    row_count: Mapped[int] = mapped_column(Integer)
    column_count: Mapped[int] = mapped_column(Integer)
    target_column: Mapped[str] = mapped_column(String)
    problem_type: Mapped[str] = mapped_column(String)  # "classification" or "regression"
    features_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # Column names and their data types, null counts, etc.
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_analysis: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    models = relationship("TrainedModel", back_populates="dataset", cascade="all, delete-orphan")


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"))
    model_name: Mapped[str] = mapped_column(String)
    algorithm: Mapped[str] = mapped_column(String)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # Model evaluation metrics dict
    file_path: Mapped[Optional[str]] = mapped_column(String)  # Path to saved model file
    version: Mapped[int] = mapped_column(Integer)
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    status: Mapped[str] = mapped_column(String, default="PENDING")  # "PENDING", "TRAINING", "DONE", "FAILED"
    status_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    dataset = relationship("Dataset", back_populates="models")
    deployments = relationship("Deployment", back_populates="model", cascade="all, delete-orphan")


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("trained_models.id"))
    status: Mapped[str] = mapped_column(String, default="inactive")  # "active" or "inactive"
    deployed_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    prediction_count: Mapped[int] = mapped_column(Integer, default=0)
    drift_threshold: Mapped[float] = mapped_column(Float, default=0.3)
    performance_threshold: Mapped[float] = mapped_column(Float, default=0.7)

    model = relationship("TrainedModel", back_populates="deployments")
    prediction_logs = relationship("PredictionLog", back_populates="deployment", cascade="all, delete-orphan")
    drift_reports = relationship("DriftReport", back_populates="deployment", cascade="all, delete-orphan")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    deployment_id: Mapped[int] = mapped_column(Integer, ForeignKey("deployments.id"))
    input_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    prediction_output: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    actual_value: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # Populated when ground truth feedback is provided
    logged_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    deployment = relationship("Deployment", back_populates="prediction_logs")


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    deployment_id: Mapped[int] = mapped_column(Integer, ForeignKey("deployments.id"))
    drift_score: Mapped[float] = mapped_column(Float)  # Ratio of features showing drift
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # Detailed drift metrics per feature
    checked_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    has_drift: Mapped[bool] = mapped_column(Boolean, default=False)

    deployment = relationship("Deployment", back_populates="drift_reports")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_type: Mapped[str] = mapped_column(String)  # "TRAINING", "DEPLOYMENT", "DRIFT", "RETRAINING", "SYSTEM"
    message: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String)  # "INFO", "WARNING", "ERROR"
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
