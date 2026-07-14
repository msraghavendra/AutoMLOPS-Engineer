import os
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union, cast

logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

EstimatorType = Any  # alias for pipelines/estimators


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every object-dtype column, attempt to strip common currency/formatting
    characters (₹, $, £, €, commas, %) and coerce to numeric. If successful,
    the column is converted in-place so downstream type detection works correctly.
    This handles datasets like Amazon sales where prices are stored as '₹599' or
    counts as '6,531'.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(r'[₹$£€,\s%]', '', regex=True)
                .str.strip()
            )
            converted: pd.Series = pd.to_numeric(cleaned, errors='coerce')  # type: ignore[assignment]
            # Only apply conversion if majority of non-null values became numeric
            non_null = int(df[col].notna().sum())  # type: ignore[arg-type]
            if non_null > 0 and int(converted.notna().sum()) / non_null >= 0.8:  # type: ignore[arg-type]
                df[col] = converted
    return df

def _is_id_or_url_column(series: pd.Series) -> bool:
    """
    Heuristic to detect ID-like or URL-like columns that are useless for ML.
    Returns True if the column should be excluded from training.
    """
    if series.dtype == object:
        sample = series.dropna().astype(str)
        if sample.empty:
            return False
        # Check for URL patterns
        url_ratio = sample.str.contains(r'https?://|www\.', regex=True, na=False).mean()
        if url_ratio > 0.3:
            return True
        # Check for high cardinality relative to dataset size (likely an ID/key column)
        uniqueness_ratio = series.nunique() / max(len(series.dropna()), 1)
        if uniqueness_ratio > 0.7 and series.nunique() > 50:
            return True
    else:
        # Numeric high-cardinality columns that look like IDs
        uniqueness_ratio = series.nunique() / max(len(series.dropna()), 1)
        if uniqueness_ratio > 0.95 and series.nunique() > 100:
            return True
    return False


def inspect_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Analyzes DataFrame to determine target column (if not provided),
    problem type (classification vs regression), and features metadata.
    High-cardinality ID/URL columns are automatically detected and excluded.
    """
    columns = list(df.columns)
    if not columns:
        raise ValueError("Dataset has no columns")

    # Coerce formatted numeric columns (e.g. '₹599', '6,531') to actual numeric dtype
    # Must happen BEFORE problem type detection so target column is correctly classified
    df = _coerce_numeric_columns(df)

    # If no target specified, find the best candidate column (skip ID/URL columns)
    if not target_col or target_col not in df.columns:
        # Prefer the last non-ID column
        candidate = None
        for col in reversed(columns):
            if not _is_id_or_url_column(df[col]):
                candidate = col
                break
        target_col = candidate or columns[-1]

    # Deduce problem type
    # If target is object, categorical, boolean, or has fewer than 15 unique values
    target_series = df[target_col].dropna()
    unique_count = target_series.nunique()
    
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    if not is_numeric or target_series.dtype == 'bool' or unique_count < 15:
        problem_type = "classification"
    else:
        problem_type = "regression"

    # Analyze features metadata
    features_metadata = {}
    for col in columns:
        if col == target_col:
            continue
        col_series = df[col]
        null_count = int(col_series.isnull().sum())  # type: ignore
        if null_count == len(df):
            # Exclude entirely null columns from training and feature mapping
            continue
        dtype_str = str(col_series.dtype)

        # Auto-detect and mark ID/URL columns so they are excluded from training
        if _is_id_or_url_column(col_series):
            features_metadata[col] = {
                "type": "categorical",
                "null_count": null_count,
                "dtype": dtype_str,
                "stats": {"unique_count": col_series.nunique()},
                "role": "ignore",
                "explanation": "Auto-excluded: high-cardinality ID/URL column — not useful for ML."
            }
            continue
        
        if pd.api.types.is_numeric_dtype(col_series):
            feature_type = "numerical"
            min_val = float(col_series.min()) if not col_series.empty else 0.0  # type: ignore
            max_val = float(col_series.max()) if not col_series.empty else 0.0  # type: ignore
            mean_val = float(col_series.mean()) if not col_series.empty else 0.0  # type: ignore
            stats: Dict[str, Any] = {"min": min_val, "max": max_val, "mean": mean_val}
        else:
            feature_type = "categorical"
            stats = {"unique_count": col_series.nunique()}

        features_metadata[col] = {
            "type": feature_type,
            "null_count": null_count,
            "dtype": dtype_str,
            "stats": stats
        }

    return target_col, problem_type, features_metadata


def build_preprocessor(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for scaling numerical and encoding categorical variables.
    max_categories=50 on OneHotEncoder prevents memory explosion from high-cardinality columns.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    features_metadata: Dict[str, Any],
    model_save_path: str
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Performs preprocessing, trains multiple models with RandomizedSearchCV,
    evaluates them, selects the best model, and saves the pipeline.
    """
    # Determine numerical and categorical feature lists, excluding ignored columns
    numerical_cols = [col for col, meta in features_metadata.items() if meta.get("role") != "ignore" and meta["type"] == "numerical"]
    categorical_cols = [col for col, meta in features_metadata.items() if meta.get("role") != "ignore" and meta["type"] == "categorical"]

    # Coerce formatted numeric columns to actual numeric types before splitting
    df = _coerce_numeric_columns(df)

    # 1. Split features and target
    features_to_use = numerical_cols + categorical_cols
    X = df[features_to_use]
    y = df[target_col]

    # Handle missing targets
    if y.isnull().any():
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

    label_mapping = None
    unique_classes = None
    if problem_type == "classification":
        y_str = y.astype(str)
        X_train, X_test, y_train_str, y_test_str = train_test_split(X, y_str, test_size=0.2, random_state=42)
        
        # Build contiguous label mapping from y_train_str only.
        # This ensures y_train has contiguous labels starting from 0, preventing XGBoost's "Invalid classes" error.
        unique_classes = sorted(y_train_str.unique().tolist())
        label_mapping = {val: idx for idx, val in enumerate(unique_classes)}
        y_train = y_train_str.map(label_mapping)
        
        # Filter test set for unseen classes (classes not present in training data cannot be predicted anyway)
        test_mask = y_test_str.isin(label_mapping.keys())
        if not test_mask.all():
            X_test = X_test[test_mask]
            y_test_str = y_test_str[test_mask]
        y_test = y_test_str.map(label_mapping)
    else:
        # Train/Test Split for regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fallback to prevent "Found array with 0 sample(s)" errors during validation/evaluation
    if X_test.empty:
        logger.warning("Test split is empty after filtering unseen classes or split. Falling back to training set for evaluation.")
        X_test = X_train.copy()
        y_test = y_train.copy()

    # Build the Preprocessor
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    # Define algorithms & parameter distributions for tuning
    models_to_train = {}
    param_grids = {}

    if problem_type == "classification":
        models_to_train = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
        }
        param_grids = {
            "Logistic Regression": {
                "model__C": [0.01, 0.1, 1.0, 10.0]
            },
            "Random Forest": {
                "model__n_estimators": [50, 100, 150],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5]
            },
            "XGBoost": {
                "model__n_estimators": [50, 100, 150],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2]
            },
            "LightGBM": {
                "model__n_estimators": [50, 100, 150],
                "model__max_depth": [-1, 5, 10],
                "model__learning_rate": [0.01, 0.1, 0.2]
            }
        }
    else:
        models_to_train = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42),
            "LightGBM": LGBMRegressor(random_state=42, verbose=-1)
        }
        param_grids = {
            "Linear Regression": {},
            "Random Forest": {
                "model__n_estimators": [50, 100, 150],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5]
            },
            "XGBoost": {
                "model__n_estimators": [50, 100, 150],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2]
            },
            "LightGBM": {
                "model__n_estimators": [50, 100, 150],
                "model__max_depth": [-1, 5, 10],
                "model__learning_rate": [0.01, 0.1, 0.2]
            }
        }

    results: Dict[str, Any] = {}
    best_score = -float('inf')
    best_algo_name = None
    best_pipeline: Any = None

    for name, model in models_to_train.items():
        best_model_pipeline: Pipeline = Pipeline(steps=[])  # will be set below
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Randomized search cv
        grid = param_grids.get(name, {})
        n_iter = min(5, len(grid) if isinstance(grid, dict) else 1)
        # Avoid running search if grid is empty (e.g. Linear Regression)
        if not grid:
            search = pipeline.fit(X_train, y_train)
            best_model_pipeline = search
        else:
            # Determine CV viability:
            # - Regression: need at least 2 samples per fold
            # - Classification: need at least 2 samples for EVERY class AND
            #   low enough class cardinality for CV to be meaningful.
            #   With hundreds/thousands of sparse classes, CV folds will
            #   inevitably miss classes → "Invalid classes" error.
            if problem_type == "classification":
                n_unique_classes = len(unique_classes) if unique_classes else y_train.nunique()
                if n_unique_classes > 30:
                    # Too many classes for CV to work reliably — skip it
                    cv_splits = 0
                else:
                    class_counts = y_train.value_counts()
                    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
                    cv_splits = min(3, min_class_count)
            else:
                cv_splits = min(3, len(y_train))

            if cv_splits >= 2:
                # We perform cross validation
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=grid,
                    n_iter=3,
                    cv=cv_splits,
                    scoring='f1_weighted' if problem_type == "classification" else 'r2',
                    random_state=42,
                    n_jobs=1
                )
                search.fit(X_train, y_train)
                best_model_pipeline = cast(Pipeline, search.best_estimator_)
            else:
                # Fall back: direct fit on training set (no CV)
                pipeline.fit(X_train, y_train)
                best_model_pipeline = pipeline

        # Evaluate on test set
        preds = best_model_pipeline.predict(X_test)
        
        metrics: Dict[str, Any] = {}
        if problem_type == "classification":
            # Add label mapper to metrics to decode predictions later
            metrics["label_mapping"] = label_mapping
            
            # Simple metrics
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average='weighted', zero_division=0))
            prec = float(precision_score(y_test, preds, average='weighted', zero_division=0))
            rec = float(recall_score(y_test, preds, average='weighted', zero_division=0))
            
            # Try computing ROC AUC
            try:
                if unique_classes and len(unique_classes) == 2:
                    probs = best_model_pipeline.predict_proba(X_test)[:, 1]
                    roc = float(roc_auc_score(y_test, probs))
                else:
                    probs = best_model_pipeline.predict_proba(X_test)
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
            comparison_metric = f1
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
            comparison_metric = r2

        results[name] = metrics

        # Select best model based on F1-score (classification) or R2-score (regression)
        if comparison_metric > best_score:
            best_score = comparison_metric
            best_algo_name = name
            best_pipeline = best_model_pipeline

    # Refit best model on full dataset.
    # We must only include classes that were present in the training set (which are in label_mapping)
    if problem_type == "classification" and label_mapping:
        y_full = y.astype(str).map(label_mapping)
        valid_idx = y_full.dropna().index
        X_full = X.loc[valid_idx]
        y_full = y_full.loc[valid_idx]
        best_pipeline.fit(X_full, y_full)
    else:
        best_pipeline.fit(X, y)

    # Save best model to disk with compression to keep file sizes reasonable
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({
        "pipeline": best_pipeline,
        "problem_type": problem_type,
        "target_column": target_col,
        "features_metadata": features_metadata,
        "label_mapping": label_mapping,
        "inverse_label_mapping": {v: k for k, v in label_mapping.items()} if label_mapping else None,
        "classes": unique_classes if problem_type == "classification" else None
    }, model_save_path, compress=3)

    assert best_algo_name is not None, "No model was trained successfully"
    return results, best_algo_name, results[best_algo_name]
