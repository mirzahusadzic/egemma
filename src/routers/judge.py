import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import settings
from ..models.judge_schema import JudgeSchemaError, check_judge_schema
from ..models.summarization import SummarizationModelWrapper, SummaryInputTooLong
from ..util.rate_limiter import get_in_memory_rate_limiter

logger = logging.getLogger(__name__)

# The summarization model's wrapper, shared (initialized in server.py): one
# Gemma model, one context, one lock.
summarization_model_wrapper: SummarizationModelWrapper | None = None

# One judgment runs at a time (the model's lock) and this many more may wait. A
# request beyond that is answered 503 before it takes a thread, so judgments
# waiting on the lock cannot fill the threadpool the other routes need.
MAX_WAITING = 2
_admission = threading.BoundedSemaphore(1 + MAX_WAITING)

router = APIRouter(tags=["Judgment"])


class JudgeRequest(BaseModel):
    # `schema` is the only name the request takes: answer_schema is internal,
    # named so that it does not shadow BaseModel.schema.
    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(
        min_length=1,
        max_length=settings.JUDGE_MAX_INPUT_CHARS,
        description="The system message, as the caller writes it: no persona is added.",
    )
    input: str = Field(
        min_length=1,
        max_length=settings.JUDGE_MAX_INPUT_CHARS,
        description="The text to read.",
    )
    answer_schema: dict[str, Any] = Field(
        alias="schema",
        description="The JSON schema the answer is held to. A $ref may only "
        "point inside it.",
    )
    max_tokens: int | None = Field(
        default=None, ge=1, le=settings.JUDGE_MAX_TOKENS, description="Answer cap."
    )
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning: bool = Field(
        default=False,
        strict=True,
        description="Let Gemma think before it answers; the thought is returned.",
    )
    max_reasoning_tokens: int | None = Field(
        default=None,
        ge=1,
        le=settings.JUDGE_MAX_REASONING_TOKENS,
        description="The thought's cap; only with reasoning.",
    )

    @model_validator(mode="after")
    def _a_cap_needs_a_thought(self) -> "JudgeRequest":
        if self.max_reasoning_tokens is not None and not self.reasoning:
            raise ValueError("max_reasoning_tokens is read only with reasoning")
        return self

    @field_validator("answer_schema")
    @classmethod
    def _schema_is_held(cls, schema: dict[str, Any]) -> dict[str, Any]:
        # Checked here, before the request is admitted: a schema that is refused
        # takes no admission slot.
        check_judge_schema(schema)
        return schema


class Violation(BaseModel):
    path: str = Field(description="A JSON pointer into the answer; empty for all.")
    message: str


class Reasoning(BaseModel):
    text: str = Field(description="The thought, as the model wrote it.")
    truncated: bool = Field(
        description="True when max_reasoning_tokens ended the thought, not the model."
    )


class JudgeResponse(BaseModel):
    answer: Any = Field(description="The answer read as JSON; null when it is not.")
    text: str
    truncated: bool
    valid: bool = Field(description="The answer is JSON and meets the whole schema.")
    violations: list[Violation]
    reasoning: Reasoning | None = Field(
        description="The thought before the answer; null unless reasoning was asked."
    )
    usage: dict[str, int]
    model: str
    revision: str


@router.post(
    "/judge",
    summary="Read a text under an instruction and answer to a JSON schema",
    description="One reading with the Gemma model, held to the schema by a "
    "grammar; with reasoning, Gemma thinks first and only the answer is held. "
    "No persona and no cache: each call is a fresh sample.",
    response_model=JudgeResponse,
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.JUDGE_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.JUDGE_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def judge(request: JudgeRequest) -> JudgeResponse:
    if summarization_model_wrapper is None or summarization_model_wrapper.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Summarization model not loaded. Set SUMMARY_MODEL_ENABLED=true "
                "in .env and restart the server."
            ),
        )
    if request.reasoning and not summarization_model_wrapper.can_reason():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "The configured model cannot reason: its chat template has no "
                "thought channel. Ask without reasoning."
            ),
        )
    if not _admission.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"A judgment is running and {MAX_WAITING} more are waiting; "
                "try again when one finishes."
            ),
            headers={"Retry-After": "10"},
        )
    try:
        judgment = await run_in_threadpool(
            summarization_model_wrapper.judge,
            request.instructions,
            request.input,
            request.answer_schema,
            request.max_tokens,
            request.temperature,
            reasoning=request.reasoning,
            max_reasoning_tokens=request.max_reasoning_tokens,
        )
    except JudgeSchemaError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except SummaryInputTooLong as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(error)
        ) from error
    except Exception as error:
        logger.error("Judgment failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Judgment failed; the server log has the details.",
        ) from error
    finally:
        # run_in_threadpool is not abandoned on cancellation, so the slot is
        # given back only once the judgment's thread has finished.
        _admission.release()
    return JudgeResponse(
        answer=judgment.answer,
        text=judgment.text,
        truncated=judgment.truncated,
        valid=judgment.valid,
        violations=judgment.violations,
        reasoning=(
            None
            if judgment.reasoning is None
            else Reasoning(
                text=judgment.reasoning, truncated=judgment.reasoning_truncated
            )
        ),
        usage={
            "input_tokens": judgment.input_tokens,
            "output_tokens": judgment.output_tokens,
            "reasoning_tokens": judgment.reasoning_tokens,
        },
        model=settings.SUMMARY_MODEL_NAME,
        revision=settings.SUMMARY_MODEL_REVISION,
    )
