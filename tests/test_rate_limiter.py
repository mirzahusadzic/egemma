import asyncio

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import src.util.rate_limiter as rate_limiter
from src.util.rate_limiter import (
    _client_last_request_time,
    get_in_memory_rate_limiter,
)


def request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "client": ("127.0.0.1", 5000),
            "scheme": "http",
            "server": ("test", 80),
            "query_string": b"",
        }
    )


def test_rate_limiter_keeps_route_budgets_independent():
    _client_last_request_time.clear()
    limiter = get_in_memory_rate_limiter(rate_limit_seconds=60, rate_limit_calls=1)

    asyncio.run(limiter(request("/amr/parse")))
    asyncio.run(limiter(request("/embed/texts")))

    with pytest.raises(HTTPException) as error:
        asyncio.run(limiter(request("/amr/parse")))
    assert error.value.status_code == 429


def test_dynamic_route_ids_share_a_budget_but_methods_are_independent():
    _client_last_request_time.clear()
    limiter = get_in_memory_rate_limiter(rate_limit_seconds=60, rate_limit_calls=1)
    app = FastAPI()

    @app.get("/conversations/{conversation_id}", dependencies=[Depends(limiter)])
    @app.delete("/conversations/{conversation_id}", dependencies=[Depends(limiter)])
    def conversation(conversation_id: str):
        return {"id": conversation_id}

    with TestClient(app) as client:
        assert client.get("/conversations/first").status_code == 200
        assert client.get("/conversations/second").status_code == 429
        assert client.delete("/conversations/second").status_code == 200
        assert client.delete("/conversations/third").status_code == 429

    assert len(_client_last_request_time) == 2
    assert {key[2] for key in _client_last_request_time} == {
        "/conversations/{conversation_id}"
    }


def test_buckets_of_clients_that_have_gone_are_swept(monkeypatch):
    _client_last_request_time.clear()
    clock = [1000.0]
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rate_limiter, "_next_sweep", 0.0)
    limiter = get_in_memory_rate_limiter(rate_limit_seconds=60, rate_limit_calls=5)

    def from_client(ip):
        req = request("/amr/parse")
        req.scope["client"] = (ip, 5000)
        return req

    for k in range(200):
        asyncio.run(limiter(from_client(f"10.0.0.{k}")))
    assert len(_client_last_request_time) == 200

    clock[0] += rate_limiter._longest_window + rate_limiter._SWEEP_EVERY
    asyncio.run(limiter(from_client("10.0.1.1")))
    assert list(_client_last_request_time) == [("10.0.1.1", "POST", "/amr/parse")]
