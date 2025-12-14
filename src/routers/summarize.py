import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool

from ..config import settings
from ..models.summarization import SummarizationModelWrapper
from ..util import condense_log, is_likely_binary
from ..util.rate_limiter import get_in_memory_rate_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared model wrapper instance (initialized in server.py)
summarization_model_wrapper: SummarizationModelWrapper | None = None

# Create router
router = APIRouter(tags=["Summarization"])


@router.post(
    "/summarize",
    summary="Summarize Code, Markdown, or Log File",
    description="Upload a code, Markdown, or log file and get a summary. "
    "Log files are automatically condensed before summarization.",
    dependencies=[
        Depends(
            get_in_memory_rate_limiter(
                rate_limit_seconds=settings.SUMMARIZE_RATE_LIMIT_SECONDS,
                rate_limit_calls=settings.SUMMARIZE_RATE_LIMIT_CALLS,
            )
        ),
    ],
)
async def summarize(
    request: Request,  # Add request object to access headers
    file: Annotated[UploadFile, File(..., max_size=settings.MAX_FILE_SIZE_BYTES)],
    max_tokens: int | None = Form(
        default=None, description="Maximum number of tokens for the summary."
    ),
    temperature: float | None = Form(
        default=None, description="Temperature for the summary generation."
    ),
    persona: str | None = Form(
        default=None,
        description=(
            "Persona to use for summarization "
            "(e.g., 'developer', 'assistant', 'security_validator')."
        ),
    ),
    model_name: str | None = Form(
        default=None,
        description=(
            "Name of the model to use for summarization (e.g., 'gemini-2.5-flash')."
        ),
    ),
    enable_safety: bool | None = Form(
        default=False,
        description=(
            "Enable Gemini safety settings for content filtering "
            "(only applies to Gemini models)."
        ),
    ),
):
    logger.debug(
        f"Received /summarize request. "
        f"Content-Type: {request.headers.get('content-type')}"
    )
    if summarization_model_wrapper is None:
        raise HTTPException(
            status_code=500, detail="Summarization service not initialized."
        )

    # Check if requesting local model when it's disabled
    if model_name is None or not model_name.startswith("gemini"):
        if not settings.SUMMARY_LOCAL_ENABLED:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Local summarization is disabled. Please specify a Gemini "
                    "model (e.g., model_name='gemini-2.5-flash')."
                ),
            )
    try:
        code_bytes = await file.read()
        if is_likely_binary(code_bytes):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file appears to be binary and cannot be summarized.",
            )
        content = code_bytes.decode("utf-8")

        ext = file.filename.split(".")[-1].lower()
        language = settings.EXTENSION_TO_LANGUAGE.get(ext, "code")

        # Condense log files before summarization
        if language == "Log File":
            content = condense_log(content)

        # Determine persona name, with defaults based on language type
        if persona is None:
            if language.lower() == "markdown":
                persona_name = "assistant"
            else:
                persona_name = "developer"
        else:
            persona_name = persona

        # Use a wrapper to pass keyword arguments to the threadpool
        def do_summarize():
            return summarization_model_wrapper.summarize(
                content,
                language=language,
                persona_name=persona_name,
                max_tokens=max_tokens,
                temperature=temperature,
                model_name=model_name,
                enable_safety=enable_safety,
            )

        summary = await run_in_threadpool(do_summarize)

        return {"language": language, "summary": summary}
    except HTTPException as e:
        logger.error(
            f"Summarization failed with HTTP Exception: {e.detail} "
            f"(status: {e.status_code}) (file: {file.filename})",
            exc_info=True,
        )
        raise e
    except Exception as e:
        logger.error(
            f"An internal server error occurred during summarization: {e} "
            f"(file: {file.filename})",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        ) from e
