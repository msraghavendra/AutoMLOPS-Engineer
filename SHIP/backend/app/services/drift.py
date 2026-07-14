import json
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Evidently imports - we import locally to handle potential missing libraries safely
# or allow checking if libraries are installed
def calculate_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: Optional[str] = None,
    drift_threshold: float = 0.3
) -> Tuple[float, Dict[str, Any], bool, str]:
    """
    Computes dataset feature drift using Evidently AI.
    Compares reference_df (training data) and current_df (production logs).
    Returns (drift_score, feature_metrics, has_drift, message).
    """
    # 1. Edge case: Check if production logs are too sparse
    min_production_records = 5
    if current_df.empty or len(current_df) < min_production_records:
        msg = f"Insufficient production data to calculate drift. Logged predictions: {len(current_df)} (requires at least {min_production_records})."
        logger.info(msg)
        return 0.0, {}, False, msg

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError as e:
        logger.error(f"Evidently AI is not installed or import failed: {e}")
        return 0.0, {}, False, "Evidently AI library is not available."

    try:
        # Pre-process columns: make sure reference and current have matching feature subsets
        # Remove target column, then intersect with current_df available columns
        ref_cols = [col for col in reference_df.columns if col != target_column]
        cur_available = set(current_df.columns)
        cols_to_use = [col for col in ref_cols if col in cur_available]

        if not cols_to_use:
            return 0.0, {}, False, "No matching feature columns between reference and current data."

        # Align columns
        ref_features = reference_df[cols_to_use].copy()
        cur_features = current_df[cols_to_use].copy()

        # Handle type mismatches or potential object casting
        for col in cols_to_use:
            # Ensure both have same dtype for categorical features
            if ref_features[col].dtype == 'object' or cur_features[col].dtype == 'object':
                ref_features[col] = ref_features[col].astype(str)
                cur_features[col] = cur_features[col].astype(str)
            # Impute missing values for drift analysis to avoid report crash
            if ref_features[col].isnull().any():
                ref_fill = ref_features[col].mode().iloc[0] if ref_features[col].dtype == 'object' else ref_features[col].median()
                ref_features[col] = ref_features[col].fillna(ref_fill)
            if cur_features[col].isnull().any():
                cur_fill = ref_features[col].mode().iloc[0] if ref_features[col].dtype == 'object' else ref_features[col].median()
                cur_features[col] = cur_features[col].fillna(cur_fill)

        # Configure Evidently Report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features, current_data=cur_features)
        
        # Parse JSON output
        result_json = report.json()
        report_data = json.loads(result_json)

        # Extract drift metrics
        dataset_drift_metric = None
        data_drift_table = None
        for metric_item in report_data["metrics"]:
            if metric_item["metric"] == "DatasetDriftMetric":
                dataset_drift_metric = metric_item["result"]
            elif metric_item["metric"] == "DataDriftTable":
                data_drift_table = metric_item["result"]

        if not dataset_drift_metric or not data_drift_table:
            raise ValueError("Evidently report is missing required metrics.")

        # Drift score is the ratio of drifted features
        drift_score = float(dataset_drift_metric["share_of_drifted_columns"])
        number_of_columns = dataset_drift_metric["number_of_columns"]
        number_of_drifted_columns = dataset_drift_metric["number_of_drifted_columns"]
        
        # Feature level metrics
        feature_metrics = {}
        for feat_name, feat_data in data_drift_table["drift_by_columns"].items():
            feature_metrics[feat_name] = {
                "drift_detected": bool(feat_data["drift_detected"]),
                "drift_score": float(feat_data["drift_score"]),  # p-value or statistic
                "test_name": str(feat_data.get("stattest_name", feat_data.get("test_name", "unknown"))),
                "feature_type": str(feat_data["column_type"])
            }

        has_drift = drift_score >= drift_threshold
        msg = f"Drift check complete. {number_of_drifted_columns} of {number_of_columns} features drifted (ratio: {drift_score:.2f})."
        
        return drift_score, feature_metrics, has_drift, msg

    except Exception as e:
        logger.error(f"Error calculating Evidently data drift: {e}")
        return 0.0, {}, False, f"Error calculating drift: {str(e)}"
