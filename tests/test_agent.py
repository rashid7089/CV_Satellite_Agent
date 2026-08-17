"""
Tests for agent/tools.py.

These mock httpx at the function level -- no live backend or AWS
credentials needed. Run them now, before Gate 2, so tool logic is proven
correct before you ever wire it to the real FastAPI service or Bedrock.

    uv add --dev pytest
    uv run pytest tests/test_agent.py -v
"""

import httpx
import pytest

from agent import tools


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://backend:8000/x")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


# ---- get_prediction_statistics -------------------------------------------

def test_get_prediction_statistics_success(monkeypatch):
    payload = {"total_predictions": 125, "class_distribution": {"water": 61, "desert": 44}}
    monkeypatch.setattr(httpx, "get", lambda url, params=None, timeout=None, **kwargs: FakeResponse(200, payload))

    result = tools.get_prediction_statistics()

    assert result == payload
    assert "error" not in result


def test_get_prediction_statistics_backend_down(monkeypatch):
    def raise_connect_error(url, params=None, timeout=None, **kwargs):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "get", raise_connect_error)

    result = tools.get_prediction_statistics()

    assert result["error"] is True
    assert "Could not reach backend" in result["message"]


def test_get_prediction_statistics_http_500(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, params=None, timeout=None, **kwargs: FakeResponse(500))

    result = tools.get_prediction_statistics()

    assert result["error"] is True
    assert result["status_code"] == 500


# ---- get_prediction_history -----------------------------------------------

def test_get_prediction_history_passes_limit(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None, **kwargs):
        seen["params"] = params
        return FakeResponse(200, {"predictions": []})

    monkeypatch.setattr(httpx, "get", fake_get)
    tools.get_prediction_history(limit=3)

    assert seen["params"] == {"limit": 3}


# ---- get_prediction_by_id --------------------------------------------------

def test_get_prediction_by_id_not_found(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, params=None, timeout=None, **kwargs: FakeResponse(404))

    result = tools.get_prediction_by_id("does-not-exist")

    assert result["error"] is True
    assert result["status_code"] == 404


# ---- get_model_info ---------------------------------------------------------

def test_get_model_info_success(monkeypatch):
    payload = {"model_name": "resnet18", "version": "1.0.0", "classes": ["cloudy", "desert", "green_area", "water"]}
    monkeypatch.setattr(httpx, "get", lambda url, params=None, timeout=None, **kwargs: FakeResponse(200, payload))

    result = tools.get_model_info()

    assert result["classes"] == ["cloudy", "desert", "green_area", "water"]


# ---- classify_image ---------------------------------------------------------

def test_classify_image_requires_real_bytes():
    result = tools.classify_image("img-123")
    assert result["error"] is True
    assert "image bytes" in result["message"]


def test_classify_image_failure_does_not_fabricate(monkeypatch):
    """Grounding requirement: a failed classification must come back as an
    explicit error, never a plausible-looking fake prediction."""
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: FakeResponse(500))

    result = tools.classify_image("img-broken")

    assert result["error"] is True
    assert "predicted_class" not in result


# ---- dispatch table consistency --------------------------------------------

def test_tool_specs_match_dispatch_table():
    """Every tool advertised to the LLM must be callable, and vice versa."""
    spec_names = {spec["toolSpec"]["name"] for spec in tools.TOOL_SPECS}
    dispatch_names = set(tools.TOOL_DISPATCH.keys())

    assert spec_names == dispatch_names


def test_agent_executes_multiple_tool_steps(monkeypatch):
    from agent import agent

    class FakeBedrock:
        def __init__(self):
            self.calls = 0

        def converse(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"stopReason": "tool_use", "output": {"message": {"role": "assistant", "content": [
                    {"toolUse": {"toolUseId": "one", "name": "get_model_info", "input": {}}}
                ]}}}
            if self.calls == 2:
                return {"stopReason": "tool_use", "output": {"message": {"role": "assistant", "content": [
                    {"toolUse": {"toolUseId": "two", "name": "get_prediction_statistics", "input": {}}}
                ]}}}
            return {"stopReason": "end_turn", "output": {"message": {"role": "assistant", "content": [{"text": "Done"}]}}}

    fake = FakeBedrock()
    monkeypatch.setattr(agent, "_bedrock_client", lambda: fake)
    monkeypatch.setattr(agent, "_execute_tool", lambda name, data: {"source": name})
    answer, history = agent.run_agent_turn([{"role": "user", "content": [{"text": "Compare the model and usage"}]}])
    assert answer == "Done"
    assert fake.calls == 3
    assert len(history) == 6
