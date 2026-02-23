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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    print(f"ERROR: {str(exc)}")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc()
        },
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
    print(f"Analyze request received: dataset={dataset.filename}")
    dataset_bytes = await dataset.read()
    baseline_bytes = await baseline.read() if baseline else None

    try:
        df = read_csv_bytes(dataset_bytes)
        print(f"Dataset loaded: {len(df)} rows")
    except Exception as exc: 
        print(f"Failed to read dataset: {exc}")
        raise HTTPException(status_code=400, detail=f"Invalid dataset CSV: {exc}") from exc

    baseline_df = None
    if baseline_bytes:
        try:
            baseline_df = read_csv_bytes(baseline_bytes)
            print(f"Baseline loaded: {len(baseline_df)} rows")
        except Exception as exc:
            print(f"Failed to read baseline: {exc}")
            raise HTTPException(status_code=400, detail=f"Invalid baseline CSV: {exc}") from exc

    print("Running pipeline...")
    try:
        result = pipeline.invoke({"dataset": df, "baseline": baseline_df, "report": {}})
        report = result["report"]
    except Exception as exc:
        print(f"Pipeline error: {exc}")
        raise exc

    print("Generating AI insights...")
    try:
        report["ai_insights"] = generate_ai_insights(report)
    except Exception as exc:
        print(f"AI insights error: {exc}")
        report["ai_insights"] = {"status": "error", "reason": str(exc)}

    report["sample_columns"] = list(df.columns)
    # Handle NaNs for JSON serialization
    sample_df = df.head(20).replace({np.nan: None})
    report["sample_rows"] = sample_df.to_dict(orient="records")
    
    print("Analysis complete.")
    from fastapi.encoders import jsonable_encoder
    return JSONResponse(content=jsonable_encoder(report))
