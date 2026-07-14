import pytest
import pandas as pd
import numpy as np
from app.services.automl import inspect_dataset, build_preprocessor

def test_inspect_dataset_classification():
    # Construct synthetic classification dataframe
    data = {
        "feat_num": [1.0, 2.0, np.nan, 4.0, 5.0],
        "feat_cat": ["A", "B", "A", None, "B"],
        "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "label": ["yes", "no", "yes", "no", "yes"]
    }
    df = pd.DataFrame(data)
    
    # Run dataset inspection
    target, problem_type, metadata = inspect_dataset(df, target_col="label")
    
    assert target == "label"
    assert problem_type == "classification"
    assert "feat_num" in metadata
    assert metadata["feat_num"]["type"] == "numerical"
    assert metadata["feat_num"]["null_count"] == 1
    
    assert "feat_cat" in metadata
    assert metadata["feat_cat"]["type"] == "categorical"
    assert metadata["feat_cat"]["null_count"] == 1
    
    # We test if the all_null column is handled or skipped.
    # In our implementation, we want to skip it from features to prevent preprocessing errors,
    # or ensure we check it.
    assert "all_null" not in metadata or metadata["all_null"]["null_count"] == 5


def test_build_preprocessor():
    # Make sure build preprocessor fits and transforms without errors
    numerical_cols = ["feat_num"]
    categorical_cols = ["feat_cat"]
    
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    
    # Simple training inputs
    X_train = pd.DataFrame({
        "feat_num": [1.0, 2.0, np.nan],
        "feat_cat": ["A", "B", "A"]
    })
    
    # Fit and transform
    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_train)
    
    # We expect 3 rows, and features: 1 scaled numeric + 2 one-hot encoded categories = 3 columns
    assert transformed.shape == (3, 3)
    # Check that nulls were imputed (no NaN in transformed array)
    assert not np.isnan(transformed).any()
