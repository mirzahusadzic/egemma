"""
Streaming utilities for the Responses API.

Provides server-sent event (SSE) streaming handlers for chat completions.
"""

from .handler import generate_streaming_response

__all__ = ["generate_streaming_response"]
