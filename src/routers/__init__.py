"""
FastAPI routers for eGemma API endpoints.

Each router handles a specific API domain:
- embed: Embedding generation
- summarize: Code/document summarization
- responses: OpenAI Responses API (chat completions)
- conversations: OpenAI Conversations API
- ast_parser: AST parsing for code analysis
"""

from . import ast_parser, conversations, embed, responses, summarize

__all__ = ["embed", "summarize", "responses", "conversations", "ast_parser"]
