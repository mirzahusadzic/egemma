"""A transformer trained on the caller's corpus alone (models/corpus_gpt.py, adapted
from Andrej Karpathy's nanoGPT, MIT).

POST /corpus/gpt trains it and reads it back. Training takes a while (about
half a minute for a few thousand words on two threads), so by default the
answer is a stream of server-sent events:
  progress  {"step", "steps", "loss"}   every 25 steps and at the end
  result    the vectors (as the JSON answer below)
  error     {"detail"}
With "stream": false the answer is one JSON document when it is done. A caller
that goes away stops the training at its next step.
"""

import asyncio
import json
import logging
import threading
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..models.corpus_gpt import Cancelled, train_corpus_gpt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/corpus", tags=["Corpus"])

# Each step's logits are 32 sentences x the longest sentence x the vocabulary, so
# these two bound the memory one training can take (under 1 GB of logits).
MAX_SENTENCE_WORDS = 256
MAX_VOCABULARY = 20_000


class CorpusGPTRequest(BaseModel):
    sentences: list[Annotated[list[str], Field(max_length=MAX_SENTENCE_WORDS)]] = Field(
        min_length=1, max_length=20000
    )
    words: dict[str, str] = Field(default_factory=dict)
    positions: dict[str, list[tuple[int, int]]] = Field(default_factory=dict)
    steps: int = Field(1500, ge=1, le=20000)
    seed: int = 7
    dimensions: int = Field(64, ge=8, le=512)
    layers: int = Field(4, ge=1, le=12)
    heads: int = Field(4, ge=1, le=16)
    threads: int = Field(2, ge=1, le=8)
    stream: bool = True

    @model_validator(mode="after")
    def _vocabulary_is_bounded(self):
        vocabulary = len({word for words in self.sentences for word in words})
        if vocabulary > MAX_VOCABULARY:
            raise ValueError(
                f"The sentences use {vocabulary} distinct words; "
                f"at most {MAX_VOCABULARY} are learned in one request."
            )
        return self


def _event(name: str, data) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


@router.post("/gpt", summary="Train a small transformer on the given corpus alone")
async def corpus_gpt(body: CorpusGPTRequest, request: Request):
    arguments = body.model_dump(exclude={"stream"})
    if not body.stream:
        try:
            return await run_in_threadpool(train_corpus_gpt, **arguments)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
            ) from error

    async def events():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        gone = threading.Event()

        def progress(step, steps, loss):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("progress", {"step": step, "steps": steps, "loss": round(loss, 4)}),
            )

        def work():
            try:
                result = train_corpus_gpt(
                    **arguments, progress=progress, cancelled=gone.is_set
                )
                loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
            except Cancelled:
                loop.call_soon_threadsafe(queue.put_nowait, (None, None))
            except ValueError as error:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", {"detail": str(error)})
                )
            except Exception as error:  # noqa: BLE001 — said to the caller, then logged
                logger.exception("The corpus transformer failed")
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", {"detail": str(error)})
                )

        threading.Thread(target=work, daemon=True).start()
        try:
            while True:
                try:
                    name, data = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        gone.set()
                        return
                    continue
                if name is None:
                    return
                yield _event(name, data)
                if name in ("result", "error"):
                    return
        finally:
            gone.set()

    return StreamingResponse(events(), media_type="text/event-stream")
