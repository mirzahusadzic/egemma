# eGemma API Reference

Complete API documentation for all eGemma endpoints.

- **Base URL:** `http://localhost:8000`
- **Authentication:** Bearer token via `Authorization: Bearer <WORKBENCH_API_KEY>`

---

## Responses API (OpenAI Agent SDK)

### POST /v1/responses

The primary endpoint for agentic workflows. Fully compatible with the OpenAI Agent SDK (`@openai/agents`).

**Request Body:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model name (`gpt-oss-20b`) |
| `input` | string/array | Yes | User input (string or array of input items) |
| `instructions` | string | No | System instructions |
| `max_output_tokens` | integer | No | Max tokens to generate (default: 4096) |
| `temperature` | float | No | Sampling temperature 0.0-2.0 (default: 1.0) |
| `reasoning_effort` | string | No | Thinking depth: low, medium, high (default: high) |
| `stream` | boolean | No | Enable SSE streaming (default: false) |
| `tools` | array | No | Tool definitions for function calling |
| `tool_choice` | string/object | No | Tool selection mode |
| `previous_response_id` | string | No | Continue from previous response |

**Example (Non-streaming):**

```bash
curl -X POST "http://localhost:8000/v1/responses" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "input": "What is 2 + 2?",
    "instructions": "You are a helpful math assistant."
  }'
```

**Response:**

```json
{
  "id": "resp_abc123def456",
  "object": "response",
  "created_at": 1234567890,
  "model": "gpt-oss-20b",
  "status": "completed",
  "output": [
    {
      "id": "rs_xyz789",
      "type": "reasoning",
      "summary": [{"type": "summary_text", "text": "Simple arithmetic..."}]
    },
    {
      "id": "msg_abc123",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [{"type": "output_text", "text": "2 + 2 = 4"}]
    }
  ],
  "output_text": "2 + 2 = 4",
  "usage": {
    "input_tokens": 25,
    "output_tokens": 10,
    "total_tokens": 35
  }
}
```

**Example with Tools:**

```bash
curl -X POST "http://localhost:8000/v1/responses" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "input": "List files in /tmp",
    "tools": [{
      "type": "function",
      "name": "bash",
      "description": "Execute bash command",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {"type": "string", "description": "Command to run"}
        },
        "required": ["command"]
      }
    }]
  }'
```

**Tool Call Response:**

```json
{
  "id": "resp_abc123",
  "output": [
    {
      "id": "msg_xyz",
      "type": "function_call",
      "call_id": "call_abc123",
      "name": "bash",
      "arguments": "{\"command\":\"ls /tmp\"}",
      "status": "completed"
    }
  ],
  "output_text": "",
  "status": "completed"
}
```

### Streaming

Enable streaming with `"stream": true`:

```bash
curl -N -X POST "http://localhost:8000/v1/responses" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b", "input": "Hello!", "stream": true}'
```

**Streaming Events:**

```json
event: response.created
data: {"type": "response.created", "response_id": "resp_xxx", "response": {...}}

event: response.reasoning_summary.delta
data: {"type": "response.reasoning_summary.delta", "delta": "Thinking...", "item_id": "rs_xxx"}

event: response.output_text.delta
data: {"type": "response.output_text.delta", "delta": "Hello", "item_id": "msg_xxx"}

event: response.output_item.added
data: {"type": "response.output_item.added", "item": {...}, "output_index": 0}

event: response.completed
data: {"type": "response.completed", "response_id": "resp_xxx", "response": {...}}
```

### Output Types

| Type | Description |
|------|-------------|
| `reasoning` | Model's thinking/analysis (o-series compatible) |
| `message` | Text response from assistant |
| `function_call` | Tool/function call request |

---

## Conversations API

Manage persistent conversations for multi-turn interactions.

### POST /v1/conversations

Create a new conversation.

```bash
curl -X POST "http://localhost:8000/v1/conversations" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"title": "My Chat"}}'
```

**Response:**

```json
{
  "id": "conv_abc123def456",
  "object": "conversation",
  "created_at": 1234567890,
  "metadata": {"title": "My Chat"},
  "items": []
}
```

### GET /v1/conversations

List all conversations.

```bash
curl "http://localhost:8000/v1/conversations?limit=20" \
  -H "Authorization: Bearer $API_KEY"
```

### GET /v1/conversations/{conversation_id}

Get a conversation by ID.

```bash
curl "http://localhost:8000/v1/conversations/conv_abc123" \
  -H "Authorization: Bearer $API_KEY"
```

### DELETE /v1/conversations/{conversation_id}

Delete a conversation.

```bash
curl -X DELETE "http://localhost:8000/v1/conversations/conv_abc123" \
  -H "Authorization: Bearer $API_KEY"
```

### GET /v1/conversations/{conversation_id}/items

Get items (messages) from a conversation.

```bash
curl "http://localhost:8000/v1/conversations/conv_abc123/items?limit=100&order=asc" \
  -H "Authorization: Bearer $API_KEY"
```

### POST /v1/conversations/{conversation_id}/items

Add items to a conversation.

```bash
curl -X POST "http://localhost:8000/v1/conversations/conv_abc123/items" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"type": "message", "role": "user", "content": "Hello!"},
      {"type": "message", "role": "assistant", "content": "Hi there!"}
    ]
  }'
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

**Example:**

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@code.py" \
  -F "persona=developer"
```

**Response:**

```json
{
  "language": "Python",
  "summary": "This module implements..."
}
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
    "returns": "int"
  }],
  "classes": [{
    "name": "MyClass",
    "docstring": "Class docs",
    "base_classes": ["BaseClass"],
    "methods": [...]
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
    "path": "models/",
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
| 404 | Not found (invalid conversation ID, etc.) |
| 429 | Too many requests (rate limit exceeded) |
| 500 | Internal server error |
| 503 | Service unavailable (model not loaded) |

Error response format:

```json
{
  "detail": "Error description"
}
```
