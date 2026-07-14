import logging
import json
import pandas as pd
from typing import Dict, Any, Optional, cast
from app.config import settings

logger = logging.getLogger(__name__)

# Try importing google.generativeai. If not installed or any import error, we fallback.
try:
    import google.generativeai as genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# Current recommended Gemini model
GEMINI_MODEL = "gemini-2.0-flash"


def get_gemini_client() -> Any:
    if not HAS_GEMINI_SDK or not settings.GEMINI_API_KEY:
        return None
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model
    except Exception as e:
        logger.warning(f"Failed to configure Gemini SDK: {e}")
        return None


def _call_gemini(model: Any, prompt: str) -> Optional[str]:
    """
    Calls Gemini and returns the raw text response, or None if it fails.
    Strips markdown code fences if present.
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip ```json ... ``` wrappers Gemini sometimes adds
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]  # drop first ```json line
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return text.strip()
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Fallback heuristic analyser (used when Gemini is unavailable)
# ---------------------------------------------------------------------------

def fallback_heuristic_analyzer(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Local rule-based heuristic profiling when Gemini API is unavailable.
    """
    columns = list(df.columns)

    # Simple target deduction
    if not target_col or target_col not in df.columns:
        candidate_cols = [c for c in columns if c.lower() not in ["id", "uuid", "index", "key"]]
        target_col = candidate_cols[-1] if candidate_cols else columns[-1]

    # Deduce problem type
    target_series = df[target_col].dropna()
    unique_count = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)

    if not is_numeric or target_series.dtype == "bool" or unique_count < 15:
        problem_type = "classification"
    else:
        problem_type = "regression"

    # Profile columns
    column_analysis: Dict[str, Any] = {}
    for col in columns:
        col_series = df[col]
        dtype_str = str(col_series.dtype)
        unique_vals = col_series.nunique()

        is_id = col.lower() in ["id", "uuid", "index", "key", "pk", "fk"]
        is_high_cardinality_non_numeric = (
            not pd.api.types.is_numeric_dtype(col_series)
        ) and (unique_vals > len(df) * 0.9)

        if col == target_col:
            role = "target"
            explanation = "Identified as the model target column."
        elif is_id or is_high_cardinality_non_numeric:
            role = "ignore"
            explanation = "Useless ID or high-cardinality non-numeric identifier."
        else:
            role = "feature"
            explanation = f"Predictive feature of type {dtype_str}."

        feat_type = "numerical" if pd.api.types.is_numeric_dtype(col_series) else "categorical"

        column_analysis[col] = {
            "role": role,
            "type": feat_type,
            "explanation": explanation,
        }

    description = (
        f"Rule-based Profile: This dataset contains {len(df)} rows and {len(columns)} columns. "
        f"The AutoML engine has identified it as a {problem_type} task targeting '{target_col}'."
    )

    return {
        "description": description,
        "suggested_target": target_col,
        "suggested_problem_type": problem_type,
        "column_analysis": column_analysis,
    }


# ---------------------------------------------------------------------------
# Gemini-powered dataset profiler
# ---------------------------------------------------------------------------

def analyze_dataset_with_ai(df: pd.DataFrame, user_target: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes dataset columns, sample data, and metadata using Gemini.
    Falls back to a local heuristic analyzer if Gemini is not configured or fails.
    """
    model = get_gemini_client()
    if not model:
        logger.info("Gemini API not configured or SDK unavailable. Running heuristic analyzer.")
        return fallback_heuristic_analyzer(df, user_target)

    try:
        columns_summary = []
        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())  # type: ignore
            unique_count = int(series.nunique())  # type: ignore
            dtype = str(series.dtype)
            sample_vals = [str(v) for v in series.dropna().head(3).tolist()]
            columns_summary.append({
                "column_name": col,
                "data_type": dtype,
                "unique_values": unique_count,
                "null_count": null_count,
                "sample_values": sample_vals,
            })

        sample_rows = [{k: str(v) for k, v in row.items()} for row in df.head(5).to_dict(orient="records")]

        prompt = f"""
You are an expert MLOps dataset profiling agent. Analyze the metadata and sample rows of the dataset below.

Dataset Columns Metadata:
{json.dumps(columns_summary, indent=2)}

First 5 Sample Rows of the Dataset:
{json.dumps(sample_rows, indent=2)}

User Specified Target Column (optional): {user_target or 'None'}

Perform the following profiling tasks:
1. Domain Identification: Write a clear 2-3 sentence description of what the dataset is about.
2. Target Column Suggestion: Identify the single most logical target column to predict.
   - If the user specified a target and it is valid, use it.
   - Otherwise deduce the best target (prefer numeric columns for regression, categorical for classification).
   - NEVER suggest a column that looks like a row ID, UUID, URL, image link, or free-text blob.
3. Problem Type: Determine whether predicting the target is "classification" or "regression".
4. Column Roles Classification: For every single column, specify:
   - role: "target", "feature", or "ignore"
     * Use "ignore" for: primary keys, row IDs, URLs, image links, free text blobs, user names, timestamps, review text, or any column with near-unique values per row.
   - type: "numerical" or "categorical"
     NOTE: If a column's sample values contain currency symbols (₹, $, £, €), commas in numbers (e.g. "6,531"), or percentage signs, classify it as "numerical" even though it appears as a string.
   - explanation: A short explanation (5-10 words).

You MUST respond with a single, raw, valid JSON object ONLY matching this schema:
{{
  "description": "Domain description...",
  "suggested_target": "column_name",
  "suggested_problem_type": "classification",
  "column_analysis": {{
    "column_name": {{
      "role": "feature",
      "type": "numerical",
      "explanation": "Brief explanation..."
    }}
  }}
}}
Do NOT wrap in markdown code blocks. Return only the raw JSON string.
"""
        text = _call_gemini(model, prompt)
        if not text:
            raise ValueError("Empty response from Gemini.")

        analysis_result: Dict[str, Any] = json.loads(text)

        if "description" not in analysis_result or "suggested_target" not in analysis_result:
            raise ValueError("Required fields missing from Gemini JSON response.")

        logger.info("Successfully profiled dataset using Gemini API.")
        return analysis_result

    except Exception as e:
        logger.error(f"Gemini profiling failed: {e}. Falling back to heuristic analyzer.", exc_info=True)
        return fallback_heuristic_analyzer(df, user_target)


# ---------------------------------------------------------------------------
# Gemini-powered data quality advisor (NEW)
# ---------------------------------------------------------------------------

def analyze_data_quality_with_ai(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Uses Gemini to dynamically detect data quality issues in the uploaded dataset.
    Returns structured warnings, detected formatting problems, and recommended fixes.
    Falls back gracefully to a basic heuristic scan if Gemini is unavailable.
    """
    model = get_gemini_client()
    if not model:
        return _heuristic_quality_scan(df)

    try:
        columns_summary = []
        for col in df.columns:
            series = df[col]
            sample_vals = [str(v) for v in series.dropna().head(5).tolist()]
            columns_summary.append({
                "column_name": col,
                "data_type": str(series.dtype),
                "null_pct": round(series.isnull().mean() * 100, 1),
                "unique_count": int(series.nunique()),  # type: ignore
                "sample_values": sample_vals,
            })

        prompt = f"""
You are a data quality expert for a machine learning pipeline. Analyze the dataset schema below and identify data quality issues.

Dataset shape: {len(df)} rows x {len(df.columns)} columns

Column Metadata:
{json.dumps(columns_summary, indent=2)}

Identify ALL of the following issues if present:
1. Columns that are numeric but stored as strings due to currency symbols (₹, $, £, €) or comma-formatted numbers (e.g. "6,531").
2. Columns with extremely high cardinality (unique_count close to total row count) that are unlikely to be useful as features.
3. Columns with very high null percentage (>50%) that may need imputation or dropping.
4. Columns that look like URLs, image links, UUIDs, or free text that should be excluded from training.
5. Any other data quality concerns for ML training.

Respond ONLY with a valid JSON object in this exact schema:
{{
  "issues": [
    {{
      "column": "column_name_or_ALL",
      "issue_type": "formatted_numeric | high_cardinality | high_nulls | useless_text | other",
      "severity": "warning | error",
      "description": "Brief description of the issue",
      "recommended_fix": "Brief recommended action"
    }}
  ],
  "overall_quality_score": 85,
  "overall_summary": "One sentence summary of dataset quality."
}}
Return ONLY raw JSON, no markdown fences.
"""
        text = _call_gemini(model, prompt)
        if not text:
            raise ValueError("Empty response from Gemini.")

        result: Dict[str, Any] = json.loads(text)
        logger.info(f"Gemini data quality analysis complete: {len(result.get('issues', []))} issues found.")
        return result

    except Exception as e:
        logger.warning(f"Gemini quality analysis failed: {e}. Using heuristic scan.")
        return _heuristic_quality_scan(df)


def _heuristic_quality_scan(df: pd.DataFrame) -> Dict[str, Any]:
    """Fallback rule-based data quality scanner."""
    issues = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        null_pct = series.isnull().mean() * 100
        unique_count = series.nunique()

        # Detect formatted numeric strings
        if series.dtype == object:
            sample = series.dropna().head(20).astype(str)
            cleaned = sample.str.replace(r'[₹$£€,\s%]', '', regex=True)
            numeric_ratio = float(cast(pd.Series, pd.to_numeric(cleaned, errors='coerce')).notna().mean())  # type: ignore
            if numeric_ratio >= 0.8:
                issues.append({
                    "column": col,
                    "issue_type": "formatted_numeric",
                    "severity": "warning",
                    "description": f"Column '{col}' contains numeric values formatted as strings (e.g. currency, commas).",
                    "recommended_fix": "Automatically stripped and converted to numeric during training.",
                })

        # High cardinality non-numeric
        if series.dtype == object and unique_count > n_rows * 0.9:
            issues.append({
                "column": col,
                "issue_type": "high_cardinality",
                "severity": "warning",
                "description": f"Column '{col}' has {unique_count} unique values ({round(unique_count/n_rows*100)}% of rows) — likely an ID or free text.",
                "recommended_fix": "Mark this column as 'ignore' in feature roles.",
            })

        # High nulls
        if null_pct > 50:
            issues.append({
                "column": col,
                "issue_type": "high_nulls",
                "severity": "warning",
                "description": f"Column '{col}' is {round(null_pct, 1)}% null.",
                "recommended_fix": "Consider dropping or imputing this column.",
            })

    score = max(0, 100 - len(issues) * 10)
    return {
        "issues": issues,
        "overall_quality_score": score,
        "overall_summary": f"Found {len(issues)} data quality issue(s) across {len(df.columns)} columns.",
    }
