"""
Thin Redis caching layer for LLM/agent calls. Wraps a function so
identical (query, context) pairs don't re-hit Groq every time —
cuts cost and latency for repeated farmer questions.

Falls back to a no-op (always calls the wrapped function) if Redis
isn't reachable, so local dev without Redis still works.
"""

import hashlib
import json
import os

try:
    import redis
except ImportError:
    redis = None


def _get_client():
    if redis is None:
        return None
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis.from_url(url, socket_connect_timeout=1)
        client.ping()
        return client
    except Exception:
        return None


_client = _get_client()


def cache_key(*parts):
    raw = json.dumps(parts, sort_keys=True, default=str)
    return "agrinex:" + hashlib.sha256(raw.encode()).hexdigest()


def cached_call(fn, cache_args, ttl_seconds=1800):
    """
    fn: zero-arg callable that performs the actual (expensive) work
    cache_args: tuple/list of hashable-ish values identifying this call
    """
    if _client is None:
        return fn()

    key = cache_key(*cache_args)
    try:
        hit = _client.get(key)
        if hit is not None:
            return json.loads(hit)
    except Exception:
        return fn()

    result = fn()
    try:
        _client.setex(key, ttl_seconds, json.dumps(result, default=str))
    except Exception:
        pass
    return result
