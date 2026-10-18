# eGemma API Reference

Complete API documentation for all eGemma endpoints.

- **Base URL:** `http://localhost:8000`
- **Authentication:** Bearer token via `Authorization: Bearer <WORKBENCH_API_KEY>`

---

## Abstract Meaning Representation

### POST /amr/parse

Parse natural-language text into an attributed AMRBART candidate graph. Set
`MODEL_PROFILE=semantic` in `.env`, then start eGemma with
`uv run uvicorn src.server:app --host localhost --port 8000` to load AMRBART
together with the Matryoshka embedding model while leaving chat and
summarization models unloaded.

```bash
curl -X POST "http://localhost:8000/amr/parse" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The parser passes the tests, but changes the dependency graph."
  }'
```

The response contains:

| Field | Meaning |
|-------|---------|
| `model`, `revision` | Exact parser identity |
| `device` | Execution device used for the query |
| `input_tokens`, `output_tokens` | Complete source-token count including boundaries, and generated-token count; output is zero when inference is skipped |
| `linearized` | Indented pointer graph for valid parses; raw model output if decoding fails |
| `linearized_raw` | Exact AMRBART pointer-token return, retained without formatting |
| `penman` | Validated standard PENMAN graph, or `null` |
| `top`, `triples` | Root and ordered typed triples from that validated graph |
| `fragments` | Diagnostic relations from an invalid graph; never a substitute for validated triples |
| `decode_status`, `decode_error` | Mechanical decoding result |

The endpoint does not claim that the graph is factually true, source-aligned,
or authorized. It exposes one learned semantic parse for downstream inspection.
JSON escapes newlines inside both formatted graph strings. Use
`jq -r '.linearized'` or `jq -r '.penman'` to print their indentation.

Decoding uses PENMAN's AMR model. Lexical roles such as `:consist-of` and
`:prep-out-of` keep their direction; genuine inverses such as `:ARG0-of` are
decoded as reversed `:ARG0` edges. The full output must contain one complete
graph with concepts and relation targets. eGemma does not repair missing
structure or apply downstream semantic canonicalization.

### POST /amr/parse-batch

Parse a bounded batch sequentially, returning the same result schema for each
input in its original order:

```bash
curl -X POST "http://localhost:8000/amr/parse-batch" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"texts":["The engineer reported a failure.","The repair needs approval."]}'
```

Both AMR endpoints return HTTP 200 for these per-input results:

| `decode_status` | Meaning |
| --- | --- |
| `ok` | A complete graph was decoded; `penman`, `top`, and `triples` are available. |
| `invalid` | Inference ran but decoding failed. Raw output, the error, and diagnostic fragments are retained; `penman` and `top` are `null`, and `triples` is empty. |
| `input_too_long` | The complete source exceeds the token budget. Inference is skipped; graph strings are empty, `penman` and `top` are `null`, and triples/fragments and output tokens are empty/zero. `decode_error` states the count and limit. |

The default limits are 1–64 inputs per batch, 1–16,384 characters per input,
400 source tokens, and 1,024 generated tokens. Source tokens include the text
boundaries and the three appended graph tokens:
`<s> text </s> <AMR> <mask> </AMR>`. The source is never silently truncated;
split long inputs explicitly. The character bound is a separate request limit,
not a guarantee that the text fits the tokenizer budget.

An invalid batch shape or character bound returns HTTP 422 for the request.
Invalid graphs and token-overflow results do not discard other batch items;
an unexpected inference/runtime failure returns HTTP 500.

## Batched semantic embeddings

### POST /embed/texts

Embed 1–256 texts through the pinned local EmbeddingGemma revision. The request
selects one supported prompt and one Matryoshka dimension (128 by default; 256,
512 or 768 explicitly); the response repeats the exact model, revision,
dimension and prompt.
JSON is wire transport only—callers decide durable representation.
Each input must contain 1–16,384 characters by default. The encoder has a
2,048-token window, including any task prompt, and may truncate beyond it;
split longer inputs before embedding when full source coverage is required.

---

## Responses API (OpenAI Agent SDK)

### POST /v1/responses

The local Responses API subset used by the OpenAI Agents SDK examples in
`scripts/`. Supports text, function calls, and SSE streaming; hosted tools and
full OpenAI API parity are not provided. The `model` field labels the response;
the server uses its configured local chat model rather than loading a model
selected by each request.

**Request Body:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model name (`gpt-oss-20b`) |
| `input` | string/array | Yes | User input (string or array of input items) |
| `instructions` | string | No | System instructions |
| `max_output_tokens` | integer | No | Max tokens to generate (default: 4096) |
| `temperature` | float | No | Sampling temperature (default: 0.3) |
| `stream` | boolean | No | Enable SSE streaming (default: false) |
| `tools` | array | No | Tool definitions for function calling |
| `tool_choice` | string/object | No | Tool selection mode |
| `conversation` / `previous_response_id` | string | No | Load stored conversation history by conversation ID |

Reasoning effort is configured server-side with `CHAT_REASONING_EFFORT`
(`high` by default); per-request reasoning overrides are not implemented.
For persisted history, pass a conversation ID created by `/v1/conversations`
as `conversation` or `previous_response_id`. Returned response IDs are not a
general-purpose persisted response store.

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
  "amr": {"calls": 10, "seconds": 60},
  "summarize": {"calls": 2, "seconds": 60},
  "chat": {"calls": 100, "seconds": 60},
  "conversation": {"calls": 100, "seconds": 60}
}
```

Budgets are per client IP, HTTP method, and matched route template. Different
conversation IDs share the same route's allowance. Storage is in-process;
these limits are intended for a single worker.

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
| 401 | Unauthorized (invalid API key) |
| 403 | Missing or malformed bearer credentials |
| 404 | Not found (invalid conversation ID, etc.) |
| 422 | Request validation failed |
| 429 | Too many requests (rate limit exceeded) |
| 500 | Internal server error |
| 503 | Service unavailable (model not loaded) |

Error response format:

```json
{
  "detail": "Error description"
}
```
