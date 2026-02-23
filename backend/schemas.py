from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ProfileColumn(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_percent: float
    unique_count: int
    sample_values: List[Any]
    numeric_summary: Dict[str, Any]
    outlier_count: int


class ProfileReport(BaseModel):
    row_count: int
    column_count: int
    columns: List[ProfileColumn]


class SchemaReport(BaseModel):
    inferred: Dict[str, str]
    violations: Dict[str, Dict[str, Any]]


class DriftReport(BaseModel):
    numeric: Dict[str, Dict[str, Any]]
    categorical: Dict[str, Dict[str, Any]]
    notes: List[str]


class Recommendation(BaseModel):
    column: str
    issue: str
    recommendation: str
    severity: str


class AnalysisReport(BaseModel):
    profile: ProfileReport
    schema: SchemaReport
    drift: Optional[DriftReport]
    recommendations: List[Recommendation]
    summary: str
