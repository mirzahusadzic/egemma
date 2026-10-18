"""Authenticated local Abstract Meaning Representation endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..config import settings
from ..models.amrbart import AMRBartModelWrapper, AMRParseResult
from ..util.rate_limiter import get_in_memory_rate_limiter

logger = logging.getLogger(__name__)


class AMRParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=settings.AMRBART_MAX_INPUT_CHARS)


class AMRParseBatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=settings.AMRBART_BATCH_MAX_INPUTS)


class AMRTripleResponse(BaseModel):
    source: str
    relation: str
    target: str
    target_kind: str


class AMRFragmentResponse(BaseModel):
    source: str | None
    relation: str
    target: str | None
    target_kind: str
    token_start: int
    token_end: int
    complete: bool


class AMRParseResponse(BaseModel):
    model: str
    revision: str
    device: str
    input_tokens: int
    output_tokens: int
    linearized: str
    linearized_raw: str | None = None
    penman: str | None
    top: str | None
    triples: list[AMRTripleResponse]
    fragments: list[AMRFragmentResponse]
    decode_status: str
    decode_error: str | None = None


amrbart_model_wrapper: AMRBartModelWrapper | None = None

router = APIRouter(tags=["AMR"])


@router.post(
    "/amr/parse",
    summary="Parse text into an attributed AMR candidate",
    response_model=AMRParseResponse,
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.AMRBART_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.AMRBART_RATE_LIMIT_CALLS,
            )
        )
    ],
)
async def parse_amr(request: AMRParseRequest) -> AMRParseResult:
    if amrbart_model_wrapper is None or not amrbart_model_wrapper.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "AMRBART model not loaded. Set MODEL_PROFILE=semantic or "
                "AMRBART_MODEL_ENABLED=true in .env and restart the server."
            ),
        )

    try:
        return await run_in_threadpool(amrbart_model_wrapper.parse, request.text)
    except Exception as exc:
        logger.exception("AMRBART inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AMRBART inference failed: {exc}",
        ) from exc


@router.post(
    "/amr/parse-batch",
    summary="Parse a bounded text batch into attributed AMR attempts",
    response_model=list[AMRParseResponse],
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.AMRBART_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.AMRBART_RATE_LIMIT_CALLS,
            )
        )
    ],
)
async def parse_amr_batch(request: AMRParseBatchRequest) -> list[AMRParseResult]:
    if amrbart_model_wrapper is None or not amrbart_model_wrapper.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "AMRBART model not loaded. Set MODEL_PROFILE=semantic or "
                "AMRBART_MODEL_ENABLED=true in .env and restart the server."
            ),
        )
    if any(
        not text or len(text) > settings.AMRBART_MAX_INPUT_CHARS
        for text in request.texts
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Every AMR input must be nonempty and within the configured bound.",
        )

    try:
        return await run_in_threadpool(
            lambda: [amrbart_model_wrapper.parse(text) for text in request.texts]
        )
    except Exception as exc:
        logger.exception("AMRBART batch inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AMRBART batch inference failed: {exc}",
        ) from exc
