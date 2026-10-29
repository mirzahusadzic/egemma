"""A request-size limit enforced before the body is parsed or authenticated.

FastAPI parses a multipart or JSON body before it runs any dependency, so a
route-level check sees a request only after the whole body has been read (and a
multipart file spooled to disk), even when the bearer key is wrong. This ASGI
middleware refuses an oversized request from its Content-Length, or as soon as a
streamed body passes the limit, before any of that happens.
"""

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from ..config import settings

# Routes that take a file upload: the file limit, plus room for multipart framing
# and the form's other fields.
UPLOAD_ROUTES = frozenset({"/summarize", "/embed", "/parse-ast"})
MULTIPART_OVERHEAD = 1024 * 1024


def request_limit(path: str) -> int:
    if path in UPLOAD_ROUTES:
        return settings.MAX_FILE_SIZE_BYTES + MULTIPART_OVERHEAD
    return settings.MAX_REQUEST_BYTES


def _too_large(limit: int) -> str:
    return f"The request exceeds the {limit:,}-byte limit."


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = request_limit(scope["path"])
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > limit:
            response = JSONResponse(
                {"detail": _too_large(limit)},
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=_too_large(limit),
                    )
            return message

        await self.app(scope, limited_receive, send)
