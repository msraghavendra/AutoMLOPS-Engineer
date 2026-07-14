import os
import logging
import pandas as pd
from typing import Any, Dict, cast
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Dataset, TrainedModel, Deployment, PredictionLog
from app.services import automl
from app.services.registry import registry_service
from app.services.deployment_manager import deployment_manager
from app.services.audit_logger import log_event

logger = logging.getLogger(__name__)

def train_model_async_task(db: Session, dataset_id: int, model_id: int):
    """
    FastAPI BackgroundTask function to perform AutoML model training and evaluation asynchronously.
    """
    # 1. Fetch records
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    db_model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    
    if not db_dataset or not db_model:
        logger.error(f"Dataset ID {dataset_id} or Model ID {model_id} not found during async training.")
        return

    try:
        # Update model status to TRAINING
        db_model.status = "TRAINING"
        db_model.status_message = "Training in progress..."
        db.commit()

        log_event(
            db, 
            event_type="TRAINING", 
            message=f"Starting AutoML training for model version {db_model.version} on dataset: {db_dataset.name}.",
            severity="INFO"
        )

        # 2. Load dataset CSV
        dataset_file_path = str(db_dataset.file_path)
        if not os.path.exists(dataset_file_path):
            raise FileNotFoundError(f"Dataset file missing at {dataset_file_path}")
        
        try:
            df = pd.read_csv(dataset_file_path)
        except Exception:
            df = pd.read_csv(dataset_file_path, engine='python')

        # 3. Retrieve any prediction feedback data with ground truth
        # Find active deployment for this model's dataset if it exists
        active_deployment = db.query(Deployment)\
            .join(TrainedModel)\
            .filter(TrainedModel.dataset_id == dataset_id, Deployment.status == "active")\
            .first()

        if active_deployment:
            # Query prediction logs with ground truths
            feedback_logs = db.query(PredictionLog)\
                .filter(PredictionLog.deployment_id == active_deployment.id, PredictionLog.actual_value.isnot(None))\
                .all()
            
            if feedback_logs:
                logger.info(f"Retraining includes {len(feedback_logs)} prediction feedback logs.")
                feedback_data = []
                for log in feedback_logs:
                    row = dict(cast(Dict[str, Any], log.input_data or {}))
                    row[cast(str, db_dataset.target_column)] = log.actual_value
                    feedback_data.append(row)
                df_feedback = pd.DataFrame(feedback_data)
                
                # Align columns and concatenate
                # Cast actual_values to match df types if needed
                df = pd.concat([df, df_feedback], ignore_index=True)

        # Define file save path
        model_filename = f"model_id{db_model.id}_v{db_model.version}_{db_model.algorithm}.joblib"
        model_save_path = os.path.join(settings.MODEL_DIR, model_filename)

        # 4. Fit AutoML models and tune hyperparameters
        results, best_algo_name, best_metrics = automl.train_and_evaluate(
            df=df,
            target_col=cast(str, db_dataset.target_column),
            problem_type=cast(str, db_dataset.problem_type),
            features_metadata=cast(Dict[str, Any], db_dataset.features_metadata or {}),
            model_save_path=model_save_path
        )

        # 5. Log details to MLflow Registry
        run_name = f"run_v{db_model.version}_{db_model.algorithm}"
        # Merge metrics and details
        log_params = {
            "dataset_name": db_dataset.name,
            "dataset_rows": len(df),
            "algorithm": db_model.algorithm,
            "problem_type": db_dataset.problem_type
        }
        
        mlflow_run_id = registry_service.log_and_register_model(
            experiment_name="Ship_It_ML",
            run_name=run_name,
            model_name=cast(str, db_model.model_name),
            metrics=best_metrics,
            params=log_params,
            model_path=model_save_path
        )

        # 6. Update TrainedModel status
        db_model.status = "DONE"
        db_model.metrics = best_metrics
        db_model.file_path = model_save_path
        db_model.mlflow_run_id = mlflow_run_id
        db_model.status_message = "Training successfully completed."
        db.commit()

        log_event(
            db, 
            event_type="TRAINING", 
            message=f"Model version {db_model.version} trained successfully using {db_model.algorithm}. Best Metric Score: {best_metrics.get('f1_score') or best_metrics.get('r2_score'):.4f}",
            severity="INFO"
        )

        # 7. Evaluate for automatic redeployment/promotion
        # If there is currently an active deployment for this dataset, check if we should auto-deploy
        if active_deployment:
            active_model = db.query(TrainedModel).filter(TrainedModel.id == active_deployment.model_id).first()
            if active_model and active_model.metrics:
                metric_key = "f1_score" if db_dataset.problem_type == "classification" else "r2_score"
                active_metric = active_model.metrics.get(metric_key, -1.0)
                new_metric = best_metrics.get(metric_key, -1.0)

                logger.info(f"Comparing performance: Active Model Metric: {active_metric:.4f}, New Model Metric: {new_metric:.4f}")

                # If new model outperforms the current active model, deploy it
                if new_metric > active_metric:
                    log_event(
                        db,
                        event_type="RETRAINING",
                        message=f"Retrained model v{db_model.version} ({new_metric:.4f}) outperformed active model v{active_model.version} ({active_metric:.4f}). Auto-promoting...",
                        severity="INFO"
                    )

                    # Swap deployments
                    active_deployment.status = "inactive"
                    
                    new_deployment = Deployment(
                        model_id=db_model.id,
                        status="active",
                        drift_threshold=active_deployment.drift_threshold,
                        performance_threshold=active_deployment.performance_threshold
                    )
                    db.add(new_deployment)
                    db.commit()
                    db.refresh(new_deployment)

                    # Swap in deployment manager
                    deployment_manager.swap_active_model(cast(int, new_deployment.id), cast(int, db_model.id), cast(str, db_model.file_path or ""))
                    
                    log_event(
                        db,
                        event_type="DEPLOYMENT",
                        message=f"Model v{db_model.version} deployed dynamically as active model.",
                        severity="INFO"
                    )
                else:
                    log_event(
                        db,
                        event_type="RETRAINING",
                        message=f"Retrained model v{db_model.version} ({new_metric:.4f}) did not outperform active model v{active_model.version} ({active_metric:.4f}). Promotion skipped.",
                        severity="INFO"
                    )

    except Exception as e:
        logger.error(f"Error during async training task: {e}", exc_info=True)
        db_model.status = "FAILED"
        db_model.status_message = str(e)
        db.commit()
        
        log_event(
            db, 
            event_type="TRAINING", 
            message=f"AutoML training failed for model version {db_model.version}: {str(e)}",
            severity="ERROR"
        )


def retrain_uploaded_model_async_task(db: Session, dataset_id: int, model_id: int, temp_model_path: str):
    """
    FastAPI BackgroundTask to retrain an uploaded scikit-learn model/pipeline on a new dataset.
    """
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        r2_score, mean_absolute_error, mean_squared_error
    )
    from app.services.automl import build_preprocessor

    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    db_model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    
    if not db_dataset or not db_model:
        logger.error(f"Dataset ID {dataset_id} or Model ID {model_id} not found during custom model retraining.")
        return

    try:
        db_model.status = "TRAINING"
        db_model.status_message = "Retraining custom uploaded model..."
        db.commit()

        log_event(
            db, 
            event_type="TRAINING", 
            message=f"Starting retraining of uploaded model version {db_model.version} on dataset: {db_dataset.name}.",
            severity="INFO"
        )

        # 1. Load dataset CSV
        dataset_file_path = str(db_dataset.file_path)
        if not os.path.exists(dataset_file_path):
            raise FileNotFoundError(f"Dataset file missing at {dataset_file_path}")
        try:
            df = pd.read_csv(dataset_file_path)
        except Exception:
            df = pd.read_csv(dataset_file_path, engine='python')

        # Prepare features and target
        features_metadata = db_dataset.features_metadata or {}
        numerical_cols = [col for col, meta in features_metadata.items() if meta.get("role") != "ignore" and meta["type"] == "numerical"]
        categorical_cols = [col for col, meta in features_metadata.items() if meta.get("role") != "ignore" and meta["type"] == "categorical"]
        
        target_col = db_dataset.target_column
        problem_type = db_dataset.problem_type

        X = df[numerical_cols + categorical_cols]
        y = df[target_col]

        # Handle missing targets
        if y.isnull().any():
            valid_indices = y.dropna().index
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]

        # Train/Test Split for metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Load uploaded model
        if not os.path.exists(temp_model_path):
            raise FileNotFoundError(f"Uploaded model file missing at {temp_model_path}")
        
        loaded_obj = joblib.load(temp_model_path)
        
        # Extract pipeline if dict, or use directly
        if isinstance(loaded_obj, dict) and "pipeline" in loaded_obj:
            pipeline = loaded_obj["pipeline"]
        else:
            pipeline = loaded_obj

        # If it is not a pipeline, wrap it in our preprocessor
        if not hasattr(pipeline, "steps"):
            logger.info("Uploaded object is a raw estimator. Wrapping in preprocessor pipeline.")
            preprocessor = build_preprocessor(numerical_cols, categorical_cols)
            pipeline = SKPipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', pipeline)
            ])

        # Encode y if classification and labels are categorical
        label_mapping = None
        unique_classes = None
        if problem_type == "classification":
            y_train_str = y_train.astype(str)
            y_test_str = y_test.astype(str)
            
            # Map based only on classes present in the training split to keep integer indices contiguous
            unique_classes = sorted(y_train_str.unique().tolist())
            label_mapping = {val: idx for idx, val in enumerate(unique_classes)}
            
            y_train = y_train_str.map(label_mapping)
            
            # Filter test set for unseen classes (classes not present in training data cannot be predicted anyway)
            test_mask = y_test_str.isin(label_mapping.keys())
            if not test_mask.all():
                X_test = X_test[test_mask]
                y_test_str = y_test_str[test_mask]
            y_test = y_test_str.map(label_mapping)

        # Fallback to prevent "Found array with 0 sample(s)" errors during validation/evaluation
        if X_test.empty:
            logger.warning("Test split is empty after filtering unseen classes or split. Falling back to training set for evaluation.")
            X_test = X_train.copy()
            y_test = y_train.copy()

        # Fit model on training split to compute metrics
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Evaluate model
        metrics: Dict[str, Any] = {}
        if problem_type == "classification":
            metrics["label_mapping"] = label_mapping
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average='weighted', zero_division=0))
            prec = float(precision_score(y_test, preds, average='weighted', zero_division=0))
            rec = float(recall_score(y_test, preds, average='weighted', zero_division=0))
            
            try:
                if unique_classes and len(unique_classes) == 2:
                    probs = pipeline.predict_proba(X_test)[:, 1]
                    roc = float(roc_auc_score(y_test, probs))
                else:
                    probs = pipeline.predict_proba(X_test)
                    roc = float(roc_auc_score(y_test, probs, multi_class='ovr'))
            except Exception:
                roc = 0.0

            metrics.update({
                "accuracy": acc,
                "f1_score": f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": roc
            })
            new_metric = f1
        else:
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = float(np.sqrt(mse))

            metrics.update({
                "r2_score": r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse
            })
            new_metric = r2

        # 3. Fit pipeline on full dataset
        if problem_type == "classification" and label_mapping:
            y_full = y.astype(str).map(label_mapping)
            valid_idx = y_full.dropna().index
            X_full = X.loc[valid_idx]
            y_full = y_full.loc[valid_idx]
            pipeline.fit(X_full, y_full)
        else:
            pipeline.fit(X, y)

        # Save retrained model to disk
        model_filename = f"model_id{db_model.id}_v{db_model.version}_custom_{db_model.model_name}.joblib"
        model_save_path = os.path.join(settings.MODEL_DIR, model_filename)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump({
            "pipeline": pipeline,
            "problem_type": problem_type,
            "target_column": target_col,
            "features_metadata": features_metadata,
            "label_mapping": label_mapping,
            "classes": unique_classes if problem_type == "classification" else None
        }, model_save_path, compress=3)

        # 4. Log details to MLflow Registry
        run_name = f"run_v{db_model.version}_{db_model.algorithm.replace(' ', '_')}"
        log_params = {
            "dataset_name": db_dataset.name,
            "dataset_rows": len(df),
            "algorithm": db_model.algorithm,
            "problem_type": db_dataset.problem_type,
            "custom_upload": "true"
        }
        
        mlflow_run_id = registry_service.log_and_register_model(
            experiment_name="Ship_It_ML",
            run_name=run_name,
            model_name=cast(str, db_model.model_name),
            metrics=metrics,
            params=log_params,
            model_path=model_save_path
        )

        # 5. Update TrainedModel status
        db_model.status = "DONE"
        db_model.metrics = metrics
        db_model.file_path = model_save_path
        db_model.mlflow_run_id = mlflow_run_id
        db_model.status_message = "Custom model retraining successfully completed."
        db.commit()

        log_event(
            db, 
            event_type="TRAINING", 
            message=f"Custom model version {db_model.version} ({db_model.algorithm}) trained successfully. Metric Score: {new_metric:.4f}",
            severity="INFO"
        )

        # 6. Evaluate for automatic deployment/promotion
        active_deployment = db.query(Deployment)\
            .join(TrainedModel)\
            .filter(TrainedModel.dataset_id == dataset_id, Deployment.status == "active")\
            .first()

        if active_deployment:
            active_model = db.query(TrainedModel).filter(TrainedModel.id == active_deployment.model_id).first()
            if active_model and active_model.metrics:
                metric_key = "f1_score" if db_dataset.problem_type == "classification" else "r2_score"
                active_metric = active_model.metrics.get(metric_key, -1.0)

                logger.info(f"Comparing performance: Active Model Metric: {active_metric:.4f}, Retrained Custom Model Metric: {new_metric:.4f}")

                if new_metric > active_metric:
                    log_event(
                        db,
                        event_type="RETRAINING",
                        message=f"Retrained custom model v{db_model.version} ({new_metric:.4f}) outperformed active model v{active_model.version} ({active_metric:.4f}). Auto-promoting...",
                        severity="INFO"
                    )

                    # Swap deployments
                    active_deployment.status = "inactive"
                    
                    new_deployment = Deployment(
                        model_id=db_model.id,
                        status="active",
                        drift_threshold=active_deployment.drift_threshold,
                        performance_threshold=active_deployment.performance_threshold
                    )
                    db.add(new_deployment)
                    db.commit()
                    db.refresh(new_deployment)

                    deployment_manager.swap_active_model(cast(int, new_deployment.id), cast(int, db_model.id), cast(str, db_model.file_path or ""))
                    
                    log_event(
                        db,
                        event_type="DEPLOYMENT",
                        message=f"Custom model v{db_model.version} deployed dynamically as active model.",
                        severity="INFO"
                    )

        # Cleanup temp file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

    except Exception as e:
        logger.error(f"Error during async custom model training task: {e}", exc_info=True)
        db_model.status = "FAILED"
        db_model.status_message = str(e)
        db.commit()

        # Cleanup temp file if exists
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
            except Exception:
                pass
        
        log_event(
            db, 
            event_type="TRAINING", 
            message=f"Custom model retraining failed for version {db_model.version}: {str(e)}",
            severity="ERROR"
        )
