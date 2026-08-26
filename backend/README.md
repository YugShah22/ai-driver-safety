# Backend — FastAPI

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Test

```bash
pytest tests/ -v
```

## Structure

```
backend/
├── app/
│   ├── main.py             # FastAPI app entry point
│   ├── api/routes/         # Route handlers
│   ├── core/config.py      # Settings via pydantic-settings
│   ├── db/                 # Supabase DB client (Phase 2)
│   ├── models/             # ORM models (Phase 2)
│   ├── schemas/            # Pydantic schemas (Phase 2)
│   ├── services/           # Business logic (Phase 2)
│   └── utils/              # Helpers (Phase 2)
└── tests/
    └── test_health.py
```
