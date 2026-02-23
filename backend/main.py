from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

try:
    from backend.pipeline.graph import build_graph
    from backend.utils.csv_utils import read_csv_bytes
    from backend.utils.ai_utils import generate_ai_insights
except ImportError:
    from pipeline.graph import build_graph
    from utils.csv_utils import read_csv_bytes
    from utils.ai_utils import generate_ai_insights

load_dotenv()

app = FastAPI(title="Data Quality Guardrails")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://data-quality-guardrails.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = build_graph()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "name": "Data Quality Guardrails",
        "status": "ok",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api")
def api_root():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    dataset: UploadFile = File(...),
    baseline: Optional[UploadFile] = File(None),
):
    dataset_bytes = await dataset.read()
    baseline_bytes = await baseline.read() if baseline else None

    try:
        df = read_csv_bytes(dataset_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid dataset CSV: {exc}") from exc

    if baseline_bytes:
        try:
            baseline_df = read_csv_bytes(baseline_bytes)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid baseline CSV: {exc}") from exc
    else:
        baseline_df = None

    result = pipeline.invoke({"dataset": df, "baseline": baseline_df, "report": {}})
    report = result["report"]
    report["sample_columns"] = list(df.columns)
    report["sample_rows"] = df.head(20).to_dict(orient="records")
    report["ai_insights"] = generate_ai_insights(report)
    return JSONResponse(report)
