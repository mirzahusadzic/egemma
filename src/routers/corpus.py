"""A transformer trained on the caller's corpus alone (models/corpus_gpt.py, adapted
from Andrej Karpathy's nanoGPT, MIT).

POST /corpus/gpt trains it and reads it back. Training takes a while (about
half a minute for a few thousand words on two threads), so by default the
answer is a stream of server-sent events:
  progress  {"step", "steps", "loss"}   every 25 steps and at the end
  result    the vectors (as the JSON answer below)
  error     {"detail"}
With "stream": false the answer is one JSON document when it is done. Either way,
a caller that goes away stops the training at its next step. One network trains at
a time and two more may wait; a request beyond that is answered 503.
"""

import asyncio
import json
import logging
import threading
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..models.corpus_gpt import Cancelled, train_corpus_gpt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/corpus", tags=["Corpus"])

# Each step's logits are 32 sentences x the longest sentence x the vocabulary, so
# these two bound the memory one training can take (under 1 GB of logits).
MAX_SENTENCE_WORDS = 256
MAX_VOCABULARY = 20_000
# What one request may read back: the floats it returns (2M is about 40 MB of
# JSON), and the places whose states it averages.
MAX_OUTPUT_FLOATS = 2_000_000
MAX_PLACES = 200_000
MAX_WORD_CHARS = 256
# One network trains at a time (models/corpus_gpt.py); this many more may wait.
MAX_WAITING = 2
_admission = threading.BoundedSemaphore(1 + MAX_WAITING)

Word = Annotated[str, Field(max_length=MAX_WORD_CHARS)]


class CorpusGPTRequest(BaseModel):
    sentences: list[Annotated[list[Word], Field(max_length=MAX_SENTENCE_WORDS)]] = (
        Field(min_length=1, max_length=20000)
    )
    words: dict[str, Word] = Field(default_factory=dict)
    positions: dict[str, list[tuple[int, int]]] = Field(default_factory=dict)
    steps: int = Field(1500, ge=1, le=20000)
    seed: int = Field(7, ge=0, le=2**32 - 1)  # torch refuses seeds past 64 bits
    dimensions: int = Field(64, ge=8, le=512)
    layers: int = Field(4, ge=1, le=12)
    heads: int = Field(4, ge=1, le=16)
    threads: int = Field(2, ge=1, le=8)
    stream: bool = True

    @model_validator(mode="after")
    def _work_is_bounded(self):
        vocabulary = len({word for words in self.sentences for word in words})
        if vocabulary > MAX_VOCABULARY:
            raise ValueError(
                f"The sentences use {vocabulary} distinct words; "
                f"at most {MAX_VOCABULARY} are learned in one request."
            )
        places = sum(len(places) for places in self.positions.values())
        if places > MAX_PLACES:
            raise ValueError(
                f"The positions name {places} places; at most {MAX_PLACES} are read."
            )
        floats = self.dimensions * (len(self.words) + len(self.positions) * self.layers)
        if floats > MAX_OUTPUT_FLOATS:
            raise ValueError(
                f"The words and positions asked for come to {floats} floats; at "
                f"most {MAX_OUTPUT_FLOATS} are returned at once."
            )
        return self


def _event(name: str, data) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _start(arguments: dict):
    """Start the training in its own thread, holding an admission slot until it
    ends. Its events go to the queue: "progress", then "result", "error" (the
    request was wrong) or "failure" (the server was), or None if it was cancelled.
    Setting the event cancels it at its next step."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    gone = threading.Event()

    def put(name, data):
        loop.call_soon_threadsafe(queue.put_nowait, (name, data))

    def progress(step, steps, loss):
        put("progress", {"step": step, "steps": steps, "loss": round(loss, 4)})

    def work():
        try:
            put(
                "result",
                train_corpus_gpt(**arguments, progress=progress, cancelled=gone.is_set),
            )
        except Cancelled:
            put(None, None)
        except ValueError as error:
            put("error", {"detail": str(error)})
        except Exception:  # noqa: BLE001 — logged here; the caller is told it failed
            logger.exception("The corpus transformer failed")
            detail = "The corpus transformer failed; the server log has the details."
            put("failure", {"detail": detail})
        finally:
            _admission.release()

    threading.Thread(target=work, daemon=True).start()
    return queue, gone


async def _events(queue: asyncio.Queue, gone: threading.Event, request: Request):
    """The training's events until it ends; a caller that goes away cancels it."""
    try:
        while True:
            try:
                name, data = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    return
                continue
            if name is None:
                return
            yield name, data
            if name in ("result", "error", "failure"):
                return
    finally:
        gone.set()


@router.post("/gpt", summary="Train a small transformer on the given corpus alone")
async def corpus_gpt(body: CorpusGPTRequest, request: Request):
    if not _admission.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"A network is training and {MAX_WAITING} more are waiting; "
                "try again when one finishes."
            ),
            headers={"Retry-After": "30"},
        )
    queue, gone = _start(body.model_dump(exclude={"stream"}))

    if body.stream:

        async def stream():
            async for name, data in _events(queue, gone, request):
                yield _event("error" if name == "failure" else name, data)

        return StreamingResponse(stream(), media_type="text/event-stream")

    async for name, data in _events(queue, gone, request):
        if name == "result":
            return data
        if name == "error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=data["detail"]
            )
        if name == "failure":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=data["detail"]
            )
    raise HTTPException(status_code=499, detail="Client closed the request")
