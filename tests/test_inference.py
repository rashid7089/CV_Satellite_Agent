import io
from PIL import Image
import numpy as np
import pytest

from app.services import inference


def image_bytes() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(out, format="PNG")
    return out.getvalue()


def test_preprocess_shape_and_range():
    batch = inference._preprocess(Image.open(io.BytesIO(image_bytes())).convert("RGB"))
    assert batch.shape == (1, 128, 128, 3)
    assert batch.dtype == np.float32
    assert 0 <= batch.min() <= batch.max() <= 1


def test_invalid_image_is_rejected():
    with pytest.raises(ValueError, match="Could not decode image"):
        inference.run_inference(b"not an image")


def test_valid_prediction_with_fake_model(monkeypatch):
    class FakeModel:
        def predict(self, batch, verbose=0):
            return np.array([[0.1, 0.2, 0.6, 0.1]])
    monkeypatch.setattr(inference, "_model", FakeModel())
    monkeypatch.setattr(inference, "_labels", ["cloudy", "desert", "green_area", "water"])
    result = inference.run_inference(image_bytes())
    assert result["predicted_class"] == "green_area"
    assert result["confidence"] == 0.6


def test_redis_inference_cache_avoids_duplicate_model_call(monkeypatch):
    storage = {}
    class FakeCache:
        def get(self, key): return storage.get(key)
        def setex(self, key, seconds, value): storage[key] = value
    class FakeModel:
        calls = 0
        def predict(self, batch, verbose=0):
            self.calls += 1
            return np.array([[0.1, 0.2, 0.6, 0.1]])
    model = FakeModel()
    monkeypatch.setattr(inference, "_model", model)
    monkeypatch.setattr(inference, "_engine", "keras")
    monkeypatch.setattr(inference, "_labels", ["cloudy", "desert", "green_area", "water"])
    monkeypatch.setattr(inference, "_cache_client", lambda: FakeCache())
    first = inference.run_inference(image_bytes())
    second = inference.run_inference(image_bytes())
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert model.calls == 1
