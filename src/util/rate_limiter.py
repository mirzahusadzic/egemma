# This is a simple in-memory rate limiter. It is not suitable for production
# environments with multiple workers (e.g., gunicorn with more than one worker).
# For such cases, a more robust solution like a Redis-based rate limiter
# would be required.

from datetime import datetime, timedelta

from fastapi import HTTPException, Request, status

# In-memory storage for rate limiting. Each route has an independent budget;
# activity on AMR parsing must not consume the embedding route's allowance.
_client_last_request_time: dict[tuple[str, str, str], list[datetime]] = {}


def get_in_memory_rate_limiter(rate_limit_seconds: int = 10, rate_limit_calls: int = 5):
    async def in_memory_rate_limiter(request: Request):
        client_ip = request.client.host
        # Dynamic IDs belong to one route budget, not one bucket per URL.
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        bucket = (client_ip, request.method, route_path)
        current_time = datetime.now()

        if bucket not in _client_last_request_time:
            _client_last_request_time[bucket] = []

        # Remove old requests outside the rate limit window
        _client_last_request_time[bucket] = [
            t
            for t in _client_last_request_time[bucket]
            if current_time - t < timedelta(seconds=rate_limit_seconds)
        ]

        if len(_client_last_request_time[bucket]) >= rate_limit_calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded. Try again in {rate_limit_seconds} seconds."
                ),
            )

        _client_last_request_time[bucket].append(current_time)

    return in_memory_rate_limiter
