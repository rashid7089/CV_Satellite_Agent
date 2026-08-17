"""
Agent tools: thin HTTP wrappers around the FastAPI backend.

Contract assumed (agree this with Student 2 in the first 20 minutes,
then update BACKEND_URL / paths here if they differ):

    GET  /health
    GET  /api/v1/model
    POST /api/v1/predict            multipart/form-data, field "image"
    GET  /api/v1/predictions?limit=N
    GET  /api/v1/predictions/{id}
    GET  /api/v1/stats

Backend isn't live yet -> every function below will raise/return an error
until Student 2's endpoints exist. That's expected. Test tool logic now
with test_agent.py (mocked HTTP), wire against the real backend once
Gate 2 is hit.

classify_image takes an already-uploaded image reference (id or server
path), NOT raw image bytes. Passing binary through an LLM tool-call
argument is unreliable across providers -- the frontend uploads the file
first (or the user uploads through Open WebUI's file feature and you
get a file path back), then the agent references it by id.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000").rstrip("/")
TIMEOUT_S = float(os.environ.get("BACKEND_TIMEOUT_S", "10"))
BACKEND_USER_EMAIL = os.environ.get("BACKEND_USER_EMAIL", "")
BACKEND_USER_PASSWORD = os.environ.get("BACKEND_USER_PASSWORD", "")
_backend_token = ""


def backend_auth_headers() -> dict[str, str]:
    global _backend_token
    if _backend_token:
        return {"Authorization": f"Bearer {_backend_token}"}
    if not BACKEND_USER_EMAIL or not BACKEND_USER_PASSWORD:
        return {}
    response = httpx.post(
        f"{BACKEND_URL}/api/v1/auth/login",
        json={"email": BACKEND_USER_EMAIL, "password": BACKEND_USER_PASSWORD},
        timeout=TIMEOUT_S,
    )
    response.raise_for_status()
    _backend_token = response.json()["access_token"]
    return {"Authorization": f"Bearer {_backend_token}"}


def _get(path: str, params: dict | None = None) -> dict[str, Any]:
    try:
        resp = httpx.get(f"{BACKEND_URL}{path}", params=params, headers=backend_auth_headers(), timeout=TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        return {"error": True, "status_code": exc.response.status_code,
                "message": f"Backend returned {exc.response.status_code} for {path}"}
    except httpx.RequestError as exc:
        return {"error": True, "message": f"Could not reach backend at {path}: {exc}"}


def classify_image(image_id: str) -> dict[str, Any]:
    """Run a previously-uploaded image through the deployed CV model.

    image_id: the identifier returned when the image was uploaded
              (e.g. by the frontend upload step, or an existing
              prediction's image reference). This tool does not accept
              raw image bytes.
    """
    return {
        "error": True,
        "message": "No image bytes were provided to this tool. Attach the image in Open WebUI "
                   "or upload it through the main frontend; do not invent an image id.",
    }


def get_prediction_history(limit: int = 5) -> dict[str, Any]:
    """Retrieve the most recent N predictions from PostgreSQL."""
    return _get("/api/v1/predictions", params={"limit": limit})


def get_prediction_by_id(prediction_id: str) -> dict[str, Any]:
    """Retrieve a specific stored prediction by its id."""
    return _get(f"/api/v1/predictions/{prediction_id}")


def get_prediction_statistics() -> dict[str, Any]:
    """Retrieve aggregated prediction information (totals, class distribution)."""
    return _get("/api/v1/stats")


def get_model_info() -> dict[str, Any]:
    """Retrieve deployed model name, version, classes, and metrics."""
    return _get("/api/v1/model")


# --- Bedrock Converse API tool specs (JSON schema per tool) ---------------
# These get passed as toolConfig={"tools": TOOL_SPECS} on every Converse
# call. Names here MUST match the keys in TOOL_DISPATCH exactly.

TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "classify_image",
            "description": "Run an already-uploaded image through the deployed "
                            "computer vision model and return predicted class, "
                            "confidence, top-K predictions, and inference latency.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string",
                                 "description": "Identifier of a previously uploaded image."}
                },
                "required": ["image_id"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "get_prediction_history",
            "description": "Get the most recent N predictions from the database.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many recent predictions to return.", "default": 5}
                },
            }},
        }
    },
    {
        "toolSpec": {
            "name": "get_prediction_by_id",
            "description": "Get one specific stored prediction by its id.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {
                    "prediction_id": {"type": "string", "description": "The prediction's database id."}
                },
                "required": ["prediction_id"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "get_prediction_statistics",
            "description": "Get aggregated stats: total predictions and class distribution.",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "get_model_info",
            "description": "Get deployed model name, version, class list, input size, and metrics.",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
]

TOOL_DISPATCH = {
    "classify_image": classify_image,
    "get_prediction_history": get_prediction_history,
    "get_prediction_by_id": get_prediction_by_id,
    "get_prediction_statistics": get_prediction_statistics,
    "get_model_info": get_model_info,
}
