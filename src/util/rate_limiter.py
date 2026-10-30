# This is a simple in-memory rate limiter. It is not suitable for production
# environments with multiple workers (e.g., gunicorn with more than one worker).
# For such cases, a more robust solution like a Redis-based rate limiter
# would be required.

import time

from fastapi import HTTPException, Request, status

# In-memory storage for rate limiting. Each route has an independent budget;
# activity on AMR parsing must not consume the embedding route's allowance.
# Times are time.monotonic() readings, so a change of the wall clock moves no window.
_client_last_request_time: dict[tuple[str, str, str], list[float]] = {}

# Buckets whose requests are all older than the longest window are swept away, at
# most once a minute, so clients that have gone do not stay in memory.
_SWEEP_EVERY = 60.0
_longest_window = 0.0
_next_sweep = 0.0


def _sweep(now: float) -> None:
    for bucket, times in list(_client_last_request_time.items()):
        if not times or now - times[-1] >= _longest_window:
            del _client_last_request_time[bucket]


def get_in_memory_rate_limiter(rate_limit_seconds: int = 10, rate_limit_calls: int = 5):
    global _longest_window
    _longest_window = max(_longest_window, rate_limit_seconds)

    async def in_memory_rate_limiter(request: Request):
        global _next_sweep
        client_ip = request.client.host
        # Dynamic IDs belong to one route budget, not one bucket per URL.
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        bucket = (client_ip, request.method, route_path)
        now = time.monotonic()

        if now >= _next_sweep:
            _sweep(now)
            _next_sweep = now + _SWEEP_EVERY

        # Remove old requests outside the rate limit window
        recent = [
            t
            for t in _client_last_request_time.get(bucket, [])
            if now - t < rate_limit_seconds
        ]
        _client_last_request_time[bucket] = recent

        if len(recent) >= rate_limit_calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded. Try again in {rate_limit_seconds} seconds."
                ),
            )

        recent.append(now)

    return in_memory_rate_limiter
