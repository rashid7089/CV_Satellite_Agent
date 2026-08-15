"""OpenAI-compatible HTTP adapter for the Bedrock tool-calling agent."""

from __future__ import annotations

import json
import os
import time
import uuid
import base64
import binascii

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.agent import MODEL_ID, run_agent_turn

PUBLIC_MODEL_ID = os.environ.get("AGENT_MODEL_NAME") or "satellite-cv-agent"
API_KEY = os.environ.get("AGENT_API_KEY") or "local-agent-key"
BACKEND_URL = (os.environ.get("BACKEND_URL") or "http://backend:8000").rstrip("/")

app = FastAPI(title="Satellite CV Bedrock Agent", version="1.0.0")


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = ""


class ChatRequest(BaseModel):
    model: str = PUBLIC_MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, alias="max_completion_tokens")


def _authorize(authorization: str | None) -> None:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid agent API key")


def _text(content: str | list | None) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
            parts.append(str(item.get("text", "")))
    return "\n".join(parts)


def _bedrock_messages(messages: list[ChatMessage]) -> list[dict]:
    converted = []
    for message in messages:
        if message.role not in {"user", "assistant"}:
            continue
        text = _text(message.content).strip()
        if text:
            converted.append({"role": message.role, "content": [{"text": text}]})
    if not converted:
        raise HTTPException(status_code=400, detail="At least one user message is required")
    return converted


def _classify_attached_images(messages: list[ChatMessage]) -> list[dict]:
    """Forward OpenAI-style image_url message parts to the CV multipart API."""
    results = []
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for item in message.content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url", {})
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str):
                continue
            try:
                if url.startswith("data:image/"):
                    header, encoded = url.split(",", 1)
                    mime = header[5:].split(";", 1)[0]
                    raw = base64.b64decode(encoded, validate=True)
                else:
                    downloaded = httpx.get(url, timeout=20, follow_redirects=True)
                    downloaded.raise_for_status()
                    raw = downloaded.content
                    mime = downloaded.headers.get("content-type", "image/jpeg").split(";", 1)[0]
                response = httpx.post(
                    f"{BACKEND_URL}/api/v1/predict",
                    files={"image": ("openwebui-upload", raw, mime)},
                    timeout=60,
                )
                response.raise_for_status()
                results.append(response.json())
            except (ValueError, binascii.Error, httpx.HTTPError) as exc:
                results.append({"error": True, "message": f"Attached image classification failed: {exc}"})
    return results


@app.get("/health")
def health():
    return {"status": "healthy", "provider": "aws-bedrock", "model": MODEL_ID}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    _authorize(authorization)
    return {"object": "list", "data": [{"id": PUBLIC_MODEL_ID, "object": "model", "owned_by": "cv-team"}]}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest, authorization: str | None = Header(default=None)):
    _authorize(authorization)
    try:
        messages = _bedrock_messages(request.messages)
        image_results = _classify_attached_images(request.messages)
        if image_results:
            verified = "VERIFIED_IMAGE_CLASSIFICATION (returned by the deployed CV API): " + json.dumps(image_results)
            if messages[-1]["role"] == "user":
                messages[-1]["content"].append({"text": verified})
            else:
                messages.append({"role": "user", "content": [{"text": verified}]})
        answer, _ = run_agent_turn(messages)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Bedrock agent failed: {exc}") from exc

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if request.stream:
        def events():
            chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                     "model": PUBLIC_MODEL_ID, "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            done = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": PUBLIC_MODEL_ID, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    return {"id": completion_id, "object": "chat.completion", "created": created,
            "model": PUBLIC_MODEL_ID, "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
