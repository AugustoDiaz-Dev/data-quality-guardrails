from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    profile = report.get("profile", {})
    columns = profile.get("columns", [])
    compact_columns: List[Dict[str, Any]] = []
    for col in columns[:50]:
        compact_columns.append({
            "name": col.get("name"),
            "dtype": col.get("dtype"),
            "missing_percent": round(float(col.get("missing_percent", 0.0)), 4),
            "unique_count": int(col.get("unique_count", 0)),
            "sample_values": col.get("sample_values", [])[:3],
            "numeric_summary": col.get("numeric_summary", {}),
            "outlier_count": int(col.get("outlier_count", 0)),
        })

    schema = report.get("schema", {})
    compact = {
        "profile": {
            "row_count": profile.get("row_count"),
            "column_count": profile.get("column_count"),
            "columns": compact_columns,
        },
        "schema": {
            "inferred": schema.get("inferred", {}),
            "violations": schema.get("violations", {}),
        },
        "drift": report.get("drift"),
        "recommendations": report.get("recommendations", [])[:20],
    }
    return compact


def generate_ai_insights(report: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "disabled", "reason": "OPENAI_API_KEY not set"}

    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"OpenAI SDK not available: {exc}"}

    default_model = "gpt-4o-mini"
    fallback_models = [
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
    ]
    configured = os.getenv("OPENAI_MODEL", default_model)
    model_candidates = [configured] + [m for m in fallback_models if m != configured]
    client = OpenAI(api_key=api_key)
    payload = _compact_report(report)
    system = (
        "You are a data quality assistant. Use only the provided report summary. "
        "Return concise, non-technical language suitable for business users."
    )
    user = {
        "task": "Generate insights in JSON only, no markdown.",
        "report": payload,
        "required_keys": [
            "summary_bullets",
            "cleaning_recipe",
            "semantic_types",
            "drift_narrative",
            "anomaly_explanation",
        ],
        "constraints": {
            "summary_bullets_max": 6,
            "cleaning_recipe_max_lines": 12,
            "semantic_types_max": 12,
            "tone": "clear, pragmatic",
        },
    }

    last_error = None
    for model in model_candidates:
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
            )
            text = response.output_text
            data = json.loads(text)
            data["status"] = "ok"
            data["model_used"] = model
            return data
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    return {"status": "error", "reason": str(last_error) if last_error else "Unknown error"}
