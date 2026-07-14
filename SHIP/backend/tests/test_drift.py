import pytest
import pandas as pd
import numpy as np
from app.services.drift import calculate_data_drift

def test_drift_insufficient_data():
    # Setup mock reference data
    ref_data = {
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
    }
    ref_df = pd.DataFrame(ref_data)
    
    # Setup current data with less than 5 rows
    current_data = {
        "feature_1": [1.0, 2.0, 3.0],
        "feature_2": ["A", "B", "A"]
    }
    current_df = pd.DataFrame(current_data)
    
    drift_score, metrics, has_drift, message = calculate_data_drift(
        reference_df=ref_df,
        current_df=current_df,
        drift_threshold=0.3
    )
    
    # Since records are < 5, drift should not be evaluated
    assert drift_score == 0.0
    assert metrics == {}
    assert has_drift is False
    assert "Insufficient production data" in message


def test_drift_calculation():
    # Setup mock reference data
    ref_data = {
        "feature_1": [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3]*5, # Mean approx 2
        "feature_2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]*5
    }
    ref_df = pd.DataFrame(ref_data)
    
    # Setup current data with drift (feature_1 values shifted significantly)
    current_data = {
        "feature_1": [10.0, 20.0, 15.0, 25.0, 18.0, 22.0, 19.0, 21.0, 17.0, 23.0]*5, # Mean approx 20
        "feature_2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]*5
    }
    current_df = pd.DataFrame(current_data)
    
    drift_score, metrics, has_drift, message = calculate_data_drift(
        reference_df=ref_df,
        current_df=current_df,
        drift_threshold=0.3
    )
    
    # Since feature_1 has completely shifted and feature_2 is identical:
    # Drift ratio should be around 0.5 (1 out of 2 features drifted)
    assert drift_score > 0.0
    assert "feature_1" in metrics
    assert "feature_2" in metrics
    # feature_1 should show drift
    assert metrics["feature_1"]["drift_detected"] is True
    # feature_2 should not show drift
    assert metrics["feature_2"]["drift_detected"] is False
    # Since 50% columns drifted and threshold is 30%, has_drift should be True
    assert has_drift is True
    assert "Drift check complete" in message
