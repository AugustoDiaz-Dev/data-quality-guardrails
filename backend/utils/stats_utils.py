from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_percent: float
    unique_count: int
    sample_values: List[Any]
    numeric_summary: Dict[str, Any]
    outlier_count: int


def _safe_sample(values: pd.Series, k: int = 5) -> List[Any]:
    unique = values.dropna().unique().tolist()
    return unique[:k]


def _iqr_outliers(series: pd.Series) -> int:
    if series.empty:
        return 0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    profiles: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_percent = float(missing_count / max(len(series), 1))
        unique_count = int(series.nunique(dropna=True))
        dtype = str(series.dtype)
        numeric_summary: Dict[str, Any] = {}
        outlier_count = 0
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_summary = {
                "min": float(np.nanmin(numeric_series)) if numeric_series.notna().any() else None,
                "max": float(np.nanmax(numeric_series)) if numeric_series.notna().any() else None,
                "mean": float(np.nanmean(numeric_series)) if numeric_series.notna().any() else None,
                "std": float(np.nanstd(numeric_series)) if numeric_series.notna().any() else None,
            }
            outlier_count = _iqr_outliers(numeric_series.dropna())
        profiles.append(
            ColumnProfile(
                name=col,
                dtype=dtype,
                missing_count=missing_count,
                missing_percent=missing_percent,
                unique_count=unique_count,
                sample_values=_safe_sample(series),
                numeric_summary=numeric_summary,
                outlier_count=outlier_count,
            ).__dict__
        )

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": profiles,
    }


def infer_column_type(series: pd.Series) -> Tuple[str, Dict[str, Any]]:
    """Infer a coarse type from the series and return type + diagnostics."""
    diagnostics: Dict[str, Any] = {}
    if pd.api.types.is_bool_dtype(series):
        return "boolean", diagnostics

    if pd.api.types.is_numeric_dtype(series):
        return "numeric", diagnostics

    non_null = series.dropna()
    if non_null.empty:
        return "string", diagnostics

    numeric_coerce = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric_coerce.notna().mean())

    datetime_coerce = pd.to_datetime(non_null, errors="coerce", utc=False)
    datetime_ratio = float(datetime_coerce.notna().mean())

    diagnostics["numeric_ratio"] = numeric_ratio
    diagnostics["datetime_ratio"] = datetime_ratio

    if datetime_ratio >= 0.9:
        return "datetime", diagnostics
    if numeric_ratio >= 0.9:
        return "numeric", diagnostics

    lowered = non_null.astype(str).str.lower()
    bool_set = {"true", "false", "0", "1", "yes", "no"}
    bool_ratio = float(lowered.isin(bool_set).mean())
    diagnostics["boolean_ratio"] = bool_ratio
    if bool_ratio >= 0.9:
        return "boolean", diagnostics

    return "string", diagnostics


def detect_schema_violations(df: pd.DataFrame, inferred_schema: Dict[str, str]) -> Dict[str, Any]:
    violations: Dict[str, Any] = {}
    for col, col_type in inferred_schema.items():
        series = df[col]
        if col_type == "numeric":
            coerced = pd.to_numeric(series, errors="coerce")
            invalid = series.notna() & coerced.isna()
            violations[col] = {
                "invalid_count": int(invalid.sum()),
                "invalid_percent": float(invalid.mean()),
            }
        elif col_type == "datetime":
            coerced = pd.to_datetime(series, errors="coerce", utc=False)
            invalid = series.notna() & coerced.isna()
            violations[col] = {
                "invalid_count": int(invalid.sum()),
                "invalid_percent": float(invalid.mean()),
            }
        elif col_type == "boolean":
            lowered = series.dropna().astype(str).str.lower()
            invalid_count = int((~lowered.isin({"true", "false", "0", "1", "yes", "no"})).sum())
            violations[col] = {
                "invalid_count": invalid_count,
                "invalid_percent": float(invalid_count / max(len(series), 1)),
            }
        else:
            violations[col] = {
                "invalid_count": 0,
                "invalid_percent": 0.0,
            }
    return violations


def detect_drift(current: pd.DataFrame, baseline: pd.DataFrame) -> Dict[str, Any]:
    drift: Dict[str, Any] = {
        "numeric": {},
        "categorical": {},
        "notes": [],
    }
    shared_cols = [c for c in current.columns if c in baseline.columns]
    for col in shared_cols:
        cur = current[col]
        base = baseline[col]
        if pd.api.types.is_numeric_dtype(cur) and pd.api.types.is_numeric_dtype(base):
            cur_num = pd.to_numeric(cur, errors="coerce")
            base_num = pd.to_numeric(base, errors="coerce")
            cur_mean = float(np.nanmean(cur_num)) if cur_num.notna().any() else None
            base_mean = float(np.nanmean(base_num)) if base_num.notna().any() else None
            cur_std = float(np.nanstd(cur_num)) if cur_num.notna().any() else None
            base_std = float(np.nanstd(base_num)) if base_num.notna().any() else None
            drift["numeric"][col] = {
                "current_mean": cur_mean,
                "baseline_mean": base_mean,
                "current_std": cur_std,
                "baseline_std": base_std,
                "mean_delta": None if (cur_mean is None or base_mean is None) else cur_mean - base_mean,
            }
        else:
            cur_top = cur.astype(str).value_counts(dropna=True).head(1)
            base_top = base.astype(str).value_counts(dropna=True).head(1)
            drift["categorical"][col] = {
                "current_top": cur_top.index[0] if not cur_top.empty else None,
                "current_top_freq": float(cur_top.iloc[0]) if not cur_top.empty else None,
                "baseline_top": base_top.index[0] if not base_top.empty else None,
                "baseline_top_freq": float(base_top.iloc[0]) if not base_top.empty else None,
            }
    if not shared_cols:
        drift["notes"].append("No shared columns between current and baseline datasets.")
    return drift


def recommend_fixes(profile: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []
    for col in profile.get("columns", []):
        name = col["name"]
        missing_pct = col["missing_percent"]
        outliers = col.get("outlier_count", 0)
        if missing_pct >= 0.05:
            recommendations.append({
                "column": name,
                "issue": "missing_values",
                "recommendation": "Consider imputing missing values (mean/median for numeric, mode for categorical).",
                "severity": "medium" if missing_pct < 0.2 else "high",
            })
        if outliers and outliers > 0:
            recommendations.append({
                "column": name,
                "issue": "outliers",
                "recommendation": "Consider clipping, winsorization, or removing outliers.",
                "severity": "medium",
            })
        schema_issue = schema.get("violations", {}).get(name)
        if schema_issue and schema_issue.get("invalid_count", 0) > 0:
            recommendations.append({
                "column": name,
                "issue": "schema_violations",
                "recommendation": "Consider casting values to the inferred type or cleaning invalid entries.",
                "severity": "high" if schema_issue.get("invalid_percent", 0) > 0.1 else "medium",
            })
    return recommendations


def summarize_report(report: Dict[str, Any]) -> str:
    profile = report.get("profile", {})
    schema = report.get("schema", {})
    drift = report.get("drift", {})
    recs = report.get("recommendations", [])

    summary_parts = [
        f"Rows: {profile.get('row_count', 0)}, Columns: {profile.get('column_count', 0)}.",
        f"Schema violations: {sum(v.get('invalid_count', 0) for v in schema.get('violations', {}).values())}.",
        f"Recommendations: {len(recs)}.",
    ]
    if drift and (drift.get("numeric") or drift.get("categorical")):
        summary_parts.append("Baseline drift comparison included.")
    return " ".join(summary_parts)
