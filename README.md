# Data Quality Guardrails (MVP)

## Run backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Run frontend

Serve `frontend/index.html` with any static server, or open in a browser and proxy requests to the backend.

Example with Python:

```bash
python -m http.server --directory frontend 5173
```

The landing page is `frontend/index.html` and the dashboard is `frontend/dashboard.html`.
The frontend calls `/api/analyze`, so run the backend on the same origin or add a proxy.
When running the frontend on localhost (e.g. port 5173), it will call `http://localhost:8000` by default.

## API
- `GET /api/health`
- `POST /api/analyze` (multipart form with `dataset`, optional `baseline`)

## Demo data
Sample CSVs are available in `demo-data/` for quick testing.
