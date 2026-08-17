# Satellite CV Production Agent

A production-oriented satellite image classifier covering four classes: `cloudy`, `desert`, `green_area`, and `water`. It includes email/password accounts with private prediction history, ONNX Runtime inference, FastAPI, PostgreSQL, Redis caching, a browser frontend, an AWS Bedrock tool-calling agent, Open WebUI, Docker Compose, monitoring, and optional S3-compatible storage.

## Architecture

Browser → Nginx frontend → FastAPI → Keras model → PostgreSQL. Open WebUI → OpenAI-compatible agent adapter → AWS Bedrock → FastAPI tools. Redis provides shared rate limiting; optional S3 stores uploaded images.

See [docs/architecture.md](docs/architecture.md) and [docs/api.md](docs/api.md).

## Dataset and model

The Kaggle Satellite Image Classification dataset contains 5,631 RGB images. Details and license notes are in [data/README.md](data/README.md). The deployed model is a transfer-learned ResNet50V2 with 128×128 input. Generate evaluation metrics with `uv run python training/evaluate.py` and export ONNX with `uv run python training/export_onnx.py`.

## Local installation

1. Install `uv` and Docker.
2. Copy `.env.example` to `.env` and replace every production secret.
3. Run `uv sync --frozen` for local Python development.
4. Run `docker compose up --build` for the full stack.

Services: frontend `:3000`, Swagger `:8000/docs`, agent `:8010`, Open WebUI `:8080`.

## Environment variables

Database, model, Bedrock, CORS, JWT authentication, rate limits, S3 storage, voice, and ports are documented in `.env.example`. Users register with email/password; passwords are stored as salted scrypt hashes and protected endpoints require a short-lived bearer session. Never commit `.env`.

## API endpoints

- `GET /health`
- `POST /api/v1/auth/register`, `POST /api/v1/auth/login`, and `GET /api/v1/auth/me`
- `POST /api/v1/predict`
- `POST /api/v1/predict/batch` (1–20 files)
- `GET /api/v1/predictions` and `/predictions/{id}`
- `GET /api/v1/stats`, `/model`, and `/monitoring`

## Agent tools and Open WebUI

The agent exposes `classify_image`, prediction history/by-id/statistics, and model information. It supports multiple sequential tool calls and refuses to invent operational data. Langfuse records agent traces, Bedrock generations, token usage, latency, and individual tool calls when its three environment variables are configured. See [openwebui/README.md](openwebui/README.md) for chat and voice setup.

## Database migrations

Run `uv run alembic upgrade head`. Production deployments should use migrations before starting the backend; `create_all` remains as a sprint-friendly compatibility fallback.

## Testing and CI

Run `uv run pytest -v`. GitHub Actions executes tests with coverage and validates Compose on pushes and pull requests.

## Docker and Dokploy deployment

Create a Dokploy Compose project from this repository, configure production environment variables and domains, attach persistent volumes for PostgreSQL/Redis/Open WebUI, deploy, then execute the smoke-test checklist in [docs/demo.md](docs/demo.md).

## Team members

Add each member's name, role, and key commits before submission.

## Known limitations

Model monitoring reports operational confidence/latency rather than ground-truth drift. Browser voice support depends on browser permissions. S3 storage is disabled by default.

## Future improvements

Add labeled production feedback, automated drift alerts, blue/green model rollout, and user-level authorization.
