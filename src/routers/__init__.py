"""
FastAPI routers for eGemma API endpoints.

Each router handles a specific API domain:
- embed: Embedding generation
- corpus: A small transformer trained on the caller's corpus
- amr: Abstract Meaning Representation parsing
- summarize: Code/document summarization
- ast_parser: AST parsing for code analysis
"""

from . import amr, ast_parser, corpus, embed, summarize

__all__ = [
    "amr",
    "ast_parser",
    "corpus",
    "embed",
    "summarize",
]
