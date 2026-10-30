import logging

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..util import read_upload
from ..util.ast_parser import parse_code_to_ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["AST Parser"])


@router.post("/parse-ast")
async def parse_ast(
    request: Request,  # Add request object to access headers
    file: UploadFile,
    language: str = Form(...),
):
    """
    Deterministic AST parsing endpoint for non-native languages.
    Currently supports Python via the ast module.
    """
    logger.debug(
        f"Received /parse-ast request. "
        f"Content-Type: {request.headers.get('content-type')}"
    )

    if language != "python":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Language '{language}' not supported. "
                "Currently only 'python' is available."
            ),
        )

    content = await read_upload(file)
    try:
        code = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="The file is not UTF-8 text."
        ) from exc

    try:
        return await run_in_threadpool(parse_code_to_ast, code, language)
    except RecursionError as exc:
        # Small inputs can nest past Python's recursion limit (a long a+b+c... chain).
        raise HTTPException(
            status_code=400, detail="The code is nested too deeply to parse."
        ) from exc
