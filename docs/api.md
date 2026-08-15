# API Documentation

FastAPI service for satellite image classification with PostgreSQL persistence.

**Base URL (local):** `http://127.0.0.1:8000`
**Interactive docs:** `http://127.0.0.1:8000/docs`

`/health` sits at the root. Every other endpoint is under `/api/v1`.

---

## Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check for API, database, and model |
| POST | `/api/v1/predict` | Classify an uploaded image and store the result |
| GET | `/api/v1/predictions` | List recent predictions |
| GET | `/api/v1/predictions/{id}` | Fetch one stored prediction |
| GET | `/api/v1/stats` | Aggregate prediction statistics |
| GET | `/api/v1/model` | Deployed model metadata |

---

## GET /health

Verifies all three subsystems. The database check runs a real `SELECT 1`, so a
`healthy` value means the connection is live, not just configured.

**Response `200`**

```json
{
  "api": "healthy",
  "database": "healthy",
  "model": "loaded"
}
```

`database` returns `unhealthy` if the connection fails. `model` returns
`not_loaded` if the artifact was never loaded at startup. The endpoint itself
always returns `200` — read the field values, not the status code.

```bash
curl http://127.0.0.1:8000/health
```

---

## POST /api/v1/predict

Runs an image through the deployed classifier and persists the result.

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | yes | JPEG, PNG, or WebP. Max 10 MB. |

**Response `201`**

```json
{
  "id": 42,
  "request_id": "9f8b2c14-3d7a-4e51-b0c9-2a6f8e1d4b30",
  "image_name": "SeaLake_2690.jpg",
  "predicted_class": "sea_lake",
  "confidence": 0.9731,
  "top_k_predictions": [
    { "class_name": "sea_lake", "probability": 0.9731 },
    { "class_name": "river", "probability": 0.0182 },
    { "class_name": "forest", "probability": 0.0087 }
  ],
  "inference_ms": 18.442,
  "model_version": "1.0.0",
  "created_at": "2026-08-13T11:24:07.918342+00:00"
}
```

**Errors**

| Status | Cause |
|---|---|
| `400` | Uploaded file is empty |
| `413` | File exceeds the 10 MB limit |
| `415` | Content type is not an allowed image format |
| `422` | Bytes could not be decoded as an image (corrupted file) |
| `500` | Model inference failed |
| `503` | Database write failed |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@data/raw/SeaLake_2690.jpg"
```

---

## GET /api/v1/predictions

Returns stored predictions, newest first.

**Query parameters**

| Name | Type | Default | Notes |
|---|---|---|---|
| `limit` | int | 20 | Between 1 and 100 |
| `offset` | int | 0 | For pagination |
| `predicted_class` | string | — | Filter to a single class |

**Response `200`** — array of prediction objects (same shape as `/predict`).

```bash
curl "http://127.0.0.1:8000/api/v1/predictions?limit=5"
curl "http://127.0.0.1:8000/api/v1/predictions?predicted_class=forest"
```

---

## GET /api/v1/predictions/{id}

**Response `200`** — a single prediction object.
**Response `404`** — no record with that id.

```bash
curl http://127.0.0.1:8000/api/v1/predictions/42
```

---

## GET /api/v1/stats

Aggregates computed in SQL, not in Python.

**Response `200`**

```json
{
  "total_predictions": 125,
  "class_distribution": {
    "forest": 61,
    "sea_lake": 44,
    "river": 20
  },
  "average_confidence": 0.9012,
  "average_inference_ms": 21.338
}
```

On an empty table, `total_predictions` is `0`, `class_distribution` is `{}`, and
both averages are `null`.

```bash
curl http://127.0.0.1:8000/api/v1/stats
```

---

## GET /api/v1/model

**Response `200`**

```json
{
  "model_name": "resnet18",
  "model_version": "1.0.0",
  "classes": ["annual_crop", "forest", "residential", "river", "sea_lake"],
  "input_size": [224, 224],
  "metrics": {
    "test_accuracy": 0.94,
    "f1_score": 0.93
  },
  "status": "deployed"
}
```

`metrics` is read from `reports/model_metrics.json` and is `null` when that file
is absent. `status` is `not_loaded` if the model artifact never loaded.

```bash
curl http://127.0.0.1:8000/api/v1/model
```

---

## Data model

All predictions are stored in a single `predictions` table.

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGINT` | Identity primary key |
| `request_id` | `UUID` | Correlates a row with an API request |
| `image_name` | `TEXT` | Sanitized upload filename |
| `image_path` | `TEXT` | Nullable |
| `image_hash` | `CHAR(64)` | SHA-256 of the uploaded bytes |
| `predicted_class` | `TEXT` | Top-1 label |
| `confidence` | `REAL` | Constrained to 0–1 |
| `top_k_predictions` | `JSONB` | Ranked label list |
| `inference_ms` | `NUMERIC(10,3)` | Constrained to >= 0 |
| `model_version` | `TEXT` | Version that produced the row |
| `created_at` | `TIMESTAMPTZ` | Server-side `now()` default |

`image_path` and `image_hash` are stored but deliberately excluded from API
responses.

---

## Agent tool mapping

The five required agent tools map directly onto these endpoints — no separate
tool endpoints exist.

| Tool | Endpoint |
|---|---|
| `classify_image` | `POST /api/v1/predict` |
| `get_prediction_history` | `GET /api/v1/predictions?limit=N` |
| `get_prediction_by_id` | `GET /api/v1/predictions/{id}` |
| `get_prediction_statistics` | `GET /api/v1/stats` |
| `get_model_info` | `GET /api/v1/model` |

---

## Error format

Every error uses FastAPI's standard envelope. Tracebacks are never exposed;
full details go to the server logs.

```json
{
  "detail": "Unsupported file type 'text/plain'. Allowed: image/jpeg, image/png, image/webp"
}
```

---

## Environment variables

Set in `backend/.env`. Never committed.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy connection string (`postgresql+psycopg://...`) |
| `MODEL_PATH` | Path to the model artifact |
| `LABELS_PATH` | Path to `labels.json` |
| `MODEL_VERSION` | Version string written to every stored row |
| `MODEL_NAME` | Architecture name reported by `/api/v1/model` |
| `CORS_ORIGINS` | Comma-separated allowed origins |
| `MAX_UPLOAD_MB` | Upload size ceiling |