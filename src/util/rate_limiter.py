from datetime import datetime, timedelta

from fastapi import HTTPException, Request, status

# In-memory storage for rate limiting
_client_last_request_time = {}


def get_in_memory_rate_limiter(rate_limit_seconds: int = 10, rate_limit_calls: int = 5):
    async def in_memory_rate_limiter(request: Request):
        client_ip = request.client.host
        current_time = datetime.now()

        if client_ip not in _client_last_request_time:
            _client_last_request_time[client_ip] = []

        # Remove old requests outside the rate limit window
        _client_last_request_time[client_ip] = [
            t
            for t in _client_last_request_time[client_ip]
            if current_time - t < timedelta(seconds=rate_limit_seconds)
        ]

        if len(_client_last_request_time[client_ip]) >= rate_limit_calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded. Try again in {rate_limit_seconds} seconds."
                ),
            )

        _client_last_request_time[client_ip].append(current_time)

    return in_memory_rate_limiter
