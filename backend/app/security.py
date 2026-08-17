import hashlib
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

from app.config import settings

_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()
_redis = None


def _redis_client():
    global _redis
    if _redis is None and settings.redis_url:
        try:
            import redis
            _redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            _redis = False
    return _redis if _redis is not False else None


def enforce_rate_limit(request: Request) -> None:
    identity = request.headers.get("authorization") or (request.client.host if request.client else "unknown")
    identity = hashlib.sha256(identity.encode()).hexdigest()[:24]
    window = settings.rate_limit_window_seconds
    limit = settings.rate_limit_requests
    bucket = int(time.time() // window)
    client = _redis_client()
    if client:
        try:
            key = f"cv:rate:{identity}:{bucket}"
            count = client.incr(key)
            if count == 1:
                client.expire(key, window + 1)
            if count > limit:
                raise HTTPException(status_code=429, detail="Rate limit exceeded.")
            return
        except HTTPException:
            raise
        except Exception:
            pass

    now = time.time()
    with _lock:
        hits = _hits[identity]
        while hits and hits[0] <= now - window:
            hits.popleft()
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")
        hits.append(now)
