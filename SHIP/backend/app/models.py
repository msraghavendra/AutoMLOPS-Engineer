import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON, Text
from sqlalchemy.orm import relationship
from app.database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    file_path = Column(String)
    row_count = Column(Integer)
    column_count = Column(Integer)
    target_column = Column(String)
    problem_type = Column(String)  # "classification" or "regression"
    features_metadata = Column(JSON)  # Column names and their data types, null counts, etc.
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    models = relationship("TrainedModel", back_populates="dataset", cascade="all, delete-orphan")


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    model_name = Column(String)
    algorithm = Column(String)
    metrics = Column(JSON)  # Model evaluation metrics dict
    file_path = Column(String)  # Path to saved model file
    version = Column(Integer)
    mlflow_run_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="PENDING")  # "PENDING", "TRAINING", "DONE", "FAILED"
    status_message = Column(Text, nullable=True)

    dataset = relationship("Dataset", back_populates="models")
    deployments = relationship("Deployment", back_populates="model", cascade="all, delete-orphan")


class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("trained_models.id"))
    status = Column(String, default="inactive")  # "active" or "inactive"
    deployed_at = Column(DateTime, default=datetime.datetime.utcnow)
    prediction_count = Column(Integer, default=0)
    drift_threshold = Column(Float, default=0.3)
    performance_threshold = Column(Float, default=0.7)

    model = relationship("TrainedModel", back_populates="deployments")
    prediction_logs = relationship("PredictionLog", back_populates="deployment", cascade="all, delete-orphan")
    drift_reports = relationship("DriftReport", back_populates="deployment", cascade="all, delete-orphan")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"))
    input_data = Column(JSON)
    prediction_output = Column(JSON)
    actual_value = Column(String, nullable=True)  # Populated when ground truth feedback is provided
    logged_at = Column(DateTime, default=datetime.datetime.utcnow)

    deployment = relationship("Deployment", back_populates="prediction_logs")


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(Integer, ForeignKey("deployments.id"))
    drift_score = Column(Float)  # Ratio of features showing drift
    metrics = Column(JSON)  # Detailed drift metrics per feature
    checked_at = Column(DateTime, default=datetime.datetime.utcnow)
    has_drift = Column(Boolean, default=False)

    deployment = relationship("Deployment", back_populates="drift_reports")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String)  # "TRAINING", "DEPLOYMENT", "DRIFT", "RETRAINING", "SYSTEM"
    message = Column(Text)
    severity = Column(String)  # "INFO", "WARNING", "ERROR"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
