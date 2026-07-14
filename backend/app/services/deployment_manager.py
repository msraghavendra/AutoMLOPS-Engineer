import os
import joblib
import logging
import threading
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from app.models import Deployment, TrainedModel

logger = logging.getLogger(__name__)

class DeploymentManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._active_model_data: Optional[Dict[str, Any]] = None
        self._active_deployment_id: Optional[int] = None
        self._active_model_id: Optional[int] = None

    def swap_active_model(self, deployment_id: int, model_id: int, model_file_path: str):
        """
        Thread-safely loads a new model from disk and updates the active model pointers.
        """
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found at {model_file_path}")

        logger.info(f"Loading model from {model_file_path} for deployment {deployment_id}...")
        model_data = joblib.load(model_file_path)

        with self._lock:
            self._active_model_data = model_data
            self._active_deployment_id = deployment_id
            self._active_model_id = model_id
        
        logger.info("Model loaded successfully. Deployment swapped.")

    def get_active_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Thread-safe read of current active model metadata.
        """
        with self._lock:
            if not self._active_model_data:
                return None
            return {
                "deployment_id": self._active_deployment_id,
                "model_id": self._active_model_id,
                "problem_type": self._active_model_data["problem_type"],
                "target_column": self._active_model_data["target_column"],
                "features_metadata": self._active_model_data["features_metadata"],
                "classes": self._active_model_data.get("classes")
            }

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thread-safe inference using the active model.
        Accepts a dictionary of inputs, converts to DataFrame, preprocesses, predicts,
        and returns prediction result (with labels mapped if classification).
        """
        with self._lock:
            if not self._active_model_data:
                raise ValueError("No active model is deployed. Please deploy a model first.")
            
            model_data = self._active_model_data

        # Convert input dictionary to DataFrame
        df_input = pd.DataFrame([inputs])

        # Validate inputs match expected features metadata
        features_metadata = model_data["features_metadata"]
        missing_features = []
        for feature in features_metadata:
            if feature not in df_input.columns:
                missing_features.append(feature)
        
        if missing_features:
            raise ValueError(f"Missing required input features: {missing_features}")

        # Run pipeline prediction
        pipeline = model_data["pipeline"]
        prediction = pipeline.predict(df_input)[0]

        # Decode prediction if classification and label mapping exists
        label_mapping = model_data.get("label_mapping")
        raw_prediction = prediction
        decoded_prediction = prediction

        if model_data["problem_type"] == "classification" and label_mapping:
            # Invert the mapping
            inv_map = {idx: val for val, idx in label_mapping.items()}
            # Cast prediction to int in case it is numpy type
            pred_idx = int(prediction)
            decoded_prediction = inv_map.get(pred_idx, raw_prediction)

        return {
            "prediction": decoded_prediction,
            "raw_prediction": int(raw_prediction) if isinstance(raw_prediction, (int, float, np.integer)) else raw_prediction,
            "problem_type": model_data["problem_type"]
        }

    def load_active_deployment_on_startup(self, db: Session):
        """
        Loads the active model from database into memory. Called on app startup.
        """
        logger.info("Initializing active model deployment on startup...")
        active_deployment = db.query(Deployment).filter(Deployment.status == "active").first()
        if not active_deployment:
            logger.info("No active deployment found in database on startup.")
            return

        model = db.query(TrainedModel).filter(TrainedModel.id == active_deployment.model_id).first()
        if not model or not model.file_path:
            logger.warning(f"Active deployment references missing or invalid model ID {active_deployment.model_id}")
            return

        try:
            self.swap_active_model(active_deployment.id, model.id, model.file_path)
            logger.info(f"Successfully loaded model version {model.version} for deployment ID {active_deployment.id} on startup.")
        except Exception as e:
            logger.error(f"Failed to load active model on startup: {e}")

# Singleton deployment manager instance
deployment_manager = DeploymentManager()
