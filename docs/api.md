# eGemma API Reference

Complete API documentation for all eGemma endpoints.

**Base URL:** `http://localhost:8000`
**Authentication:** Bearer token via `Authorization: Bearer <WORKBENCH_API_KEY>`

---

## Chat Completions

### POST /v1/chat/completions

OpenAI-compatible chat completions with GPT-OSS-20B.

**Request Body:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model name (`gpt-oss-20b`) |
| `messages` | array | Yes | Array of `{role, content}` objects |
| `max_tokens` | integer | No | Max tokens to generate (default: 4096) |
| `temperature` | float | No | Sampling temperature 0.0-2.0 (default: 0.7) |
| `stream` | boolean | No | Enable SSE streaming (default: false) |
| `tools` | array | No | Tool definitions for function calling |
| `tool_choice` | string/object | No | Tool selection mode |
| `include_thinking` | boolean | No | Include reasoning traces (default: false) |

**Example:**

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing."}
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

**Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-oss-20b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Quantum computing uses qubits...",
      "tool_calls": null
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  }
}
```

**Streaming:**

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b", "messages": [...], "stream": true}'
```

Returns Server-Sent Events (SSE) with `data: {chunk}` format.

**Extended Thinking:**

When `include_thinking: true`, responses include reasoning traces:

```json
{
  "choices": [{
    "message": {
      "content": "2 + 2 = 4",
      "thinking": "User asks for basic arithmetic. 2 + 2 equals 4."
    }
  }]
}
```

---

## Session Management

### POST /v1/sessions

Create a new chat session.

```bash
curl -X POST "http://localhost:8000/v1/sessions" \
  -H "Authorization: Bearer $API_KEY"
```

**Response:**

```json
{
  "session_id": "sess_abc123def456",
  "max_context": 65536,
  "created_at": 1234567890.123
}
```

### GET /v1/sessions

List all saved sessions (sorted by last accessed).

```bash
curl "http://localhost:8000/v1/sessions" \
  -H "Authorization: Bearer $API_KEY"
```

### GET /v1/sessions/current

Get current active session stats.

```bash
curl "http://localhost:8000/v1/sessions/current" \
  -H "Authorization: Bearer $API_KEY"
```

**Response:**

```json
{
  "session_id": "sess_abc123def456",
  "message_count": 4,
  "token_count": 256,
  "max_context": 65536,
  "tokens_remaining": 65280,
  "context_usage": 0.0039
}
```

### POST /v1/sessions/{session_id}/load

Resume a previously saved session.

```bash
curl -X POST "http://localhost:8000/v1/sessions/sess_abc123/load" \
  -H "Authorization: Bearer $API_KEY"
```

### GET /v1/sessions/{session_id}/messages

Retrieve all messages from a session.

```bash
curl "http://localhost:8000/v1/sessions/sess_abc123/messages" \
  -H "Authorization: Bearer $API_KEY"
```

### POST /v1/sessions/count-tokens

Count tokens in text using the model's tokenizer.

```bash
curl -X POST "http://localhost:8000/v1/sessions/count-tokens?text=Hello%20world" \
  -H "Authorization: Bearer $API_KEY"
```

### Using Sessions with Chat

Pass `X-Session-Id` header to enable session-based conversations:

```bash
# First message
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "X-Session-Id: sess_abc123" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "My name is Alice."}]}'

# Follow-up (history auto-prepended)
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "X-Session-Id: sess_abc123" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is my name?"}]}'
```

---

## Embeddings

### POST /embed

Generate text embeddings using Gemma 300M with Matryoshka support.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimensions` | integer | 128 | Output dimensions: 128, 256, 512, or 768 |
| `prompt_name` | string | document | Prompt type: query, document, clustering, etc. |
| `title` | string | null | Optional title for document embeddings |

**Example:**

```bash
curl -X POST "http://localhost:8000/embed?dimensions=256&prompt_name=document" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@document.txt"
```

**Response:**

```json
{
  "embedding_256d": [0.123, -0.456, ...]
}
```

---

## Summarization

### POST /summarize

Summarize code, logs, or documents using local Gemma or Gemini API.

**Form Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | File to summarize |
| `model_name` | string | null | Model override (e.g., `gemini-2.5-flash`) |
| `persona` | string | auto | Persona: developer, assistant, log_analyst, security_validator |
| `max_tokens` | integer | null | Max output tokens |
| `temperature` | float | null | Sampling temperature |
| `enable_safety` | boolean | false | Enable Gemini safety filters |

**Example (local Gemma):**

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@code.py" \
  -F "persona=developer"
```

**Example (Gemini API):**

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@code.py" \
  -F "model_name=gemini-2.5-flash" \
  -F "persona=developer"
```

**Response:**

```json
{
  "language": "Python",
  "summary": "This module implements..."
}
```

**Security Validator Persona:**

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@document.md" \
  -F "model_name=gemini-2.5-flash" \
  -F "persona=security_validator" \
  -F "enable_safety=true"
```

Response format:

```text
THREAT ASSESSMENT: [SAFE | SUSPICIOUS | MALICIOUS]
DETECTED PATTERNS: [List]
SPECIFIC CONCERNS: [Quotes]
RECOMMENDATION: [APPROVE | REVIEW | REJECT]
REASONING: [Explanation]
```

---

## AST Parsing

### POST /parse-ast

Parse Python code into structured AST representation.

**Form Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | file | Python source file |
| `language` | string | Must be `python` |

**Example:**

```bash
curl -X POST "http://localhost:8000/parse-ast" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@code.py" \
  -F "language=python"
```

**Response:**

```json
{
  "language": "python",
  "docstring": "Module docstring",
  "imports": ["os", "typing"],
  "functions": [{
    "name": "my_function",
    "docstring": "Function docs",
    "params": [{"name": "arg1", "type": "str"}],
    "returns": "int",
    "decorators": [],
    "is_async": false
  }],
  "classes": [{
    "name": "MyClass",
    "docstring": "Class docs",
    "base_classes": ["BaseClass"],
    "methods": [...],
    "decorators": []
  }]
}
```

---

## Monitoring

### GET /health

Check server and model status (no auth required).

```bash
curl "http://localhost:8000/health"
```

**Response:**

```json
{
  "status": "ok",
  "embedding_model": {"name": "google/embeddinggemma-300m", "status": "loaded"},
  "local_summarization_model": {"status": "disabled"},
  "gemini_api": {"api_key_set": true, "default_model": "gemini-2.5-flash"},
  "chat_model": {
    "name": "gpt-oss-20b",
    "path": "models/gpt-oss-20b-Q4_K_M.gguf",
    "status": "loaded",
    "context_length": 65536,
    "supports_tools": true
  }
}
```

### GET /rate-limits

Get current rate limit configuration (no auth required).

```bash
curl "http://localhost:8000/rate-limits"
```

**Response:**

```json
{
  "embed": {"calls": 5, "seconds": 10},
  "summarize": {"calls": 2, "seconds": 60},
  "chat": {"calls": 10, "seconds": 60}
}
```

---

## Models

### GET /v1/models

List available models (OpenAI-compatible).

```bash
curl "http://localhost:8000/v1/models"
```

**Response:**

```json
{
  "object": "list",
  "data": [{
    "id": "gpt-oss-20b",
    "object": "model",
    "created": 1234567890,
    "owned_by": "egemma"
  }]
}
```

---

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (invalid/missing API key) |
| 404 | Not found (invalid session ID, etc.) |
| 429 | Too many requests (rate limit exceeded) |
| 500 | Internal server error |
| 503 | Service unavailable (model not loaded) |

Error response format:

```json
{
  "detail": "Error description"
}
```
