import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
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

def inspect_dataset(df: pd.DataFrame, target_col: str = None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Analyzes DataFrame to determine target column (if not provided),
    problem type (classification vs regression), and features metadata.
    """
    columns = list(df.columns)
    if not columns:
        raise ValueError("Dataset has no columns")

    # If no target specified, use the last column
    if not target_col or target_col not in df.columns:
        target_col = columns[-1]

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
        null_count = int(col_series.isnull().sum())
        if null_count == len(df):
            # Exclude entirely null columns from training and feature mapping
            continue
        dtype_str = str(col_series.dtype)
        
        if pd.api.types.is_numeric_dtype(col_series):
            feature_type = "numerical"
            min_val = float(col_series.min()) if not col_series.empty else 0.0
            max_val = float(col_series.max()) if not col_series.empty else 0.0
            mean_val = float(col_series.mean()) if not col_series.empty else 0.0
            stats = {"min": min_val, "max": max_val, "mean": mean_val}
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
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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
    # 1. Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Determine numerical and categorical feature lists
    numerical_cols = [col for col, meta in features_metadata.items() if meta["type"] == "numerical"]
    categorical_cols = [col for col, meta in features_metadata.items() if meta["type"] == "categorical"]

    # Handle missing targets
    if y.isnull().any():
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the Preprocessor
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    # Encode y if classification and labels are categorical
    label_mapping = None
    if problem_type == "classification":
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)
        # Sort classes to ensure deterministic ordering
        unique_classes = sorted(list(set(y_train.unique()) | set(y_test.unique())))
        label_mapping = {val: idx for idx, val in enumerate(unique_classes)}
        y_train = y_train.map(label_mapping)
        y_test = y_test.map(label_mapping)

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

    results = {}
    best_score = -float('inf')
    best_algo_name = None
    best_pipeline = None

    for name, model in models_to_train.items():
        # Create pipeline with preprocessor and model
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
            # Determine dynamic CV folds based on training set sizes and cardinality
            if problem_type == "classification":
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
                best_model_pipeline = search.best_estimator_
            else:
                # Fall back to fitting baseline model on the training set
                pipeline.fit(X_train, y_train)
                best_model_pipeline = pipeline

        # Evaluate on test set
        preds = best_model_pipeline.predict(X_test)
        
        metrics = {}
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
                if len(unique_classes) == 2:
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
            r2 = float(r2_score(y_test, preds))
            mae = float(mean_absolute_error(y_test, preds))
            mse = float(mean_squared_error(y_test, preds))
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

    # Refit best model on full dataset
    y_full = y
    if problem_type == "classification" and label_mapping:
        y_full = y_full.astype(str).map(label_mapping)

    best_pipeline.fit(X, y_full)

    # Save best model to disk
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({
        "pipeline": best_pipeline,
        "problem_type": problem_type,
        "target_column": target_col,
        "features_metadata": features_metadata,
        "label_mapping": label_mapping,
        "classes": unique_classes if problem_type == "classification" else None
    }, model_save_path)

    return results, best_algo_name, results[best_algo_name]
