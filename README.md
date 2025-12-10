# eGemma - Local AI Workbench

<div align="center">
<img src="./docs/assets/egemma-logo.png" alt="eGemma Logo" width="400"/>

A unified API server for embeddings, summarization, and chat completions.
**Fully compatible with OpenAI Agent SDK** - run powerful AI models locally.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Features

| Capability | Model | Description |
|------------|-------|-------------|
| **Chat/Agents** | GPT-OSS-20B | OpenAI Agent SDK compatible, 128K context, tool calling |
| **Embeddings** | Gemma 300M | Matryoshka support (128-768 dims), optimized prompts |
| **Summarization** | Gemma 12B / Gemini API | Code, logs, docs with persona-based analysis |

**Infrastructure:** Bearer auth, rate limiting, LRU caching, Metal/CUDA acceleration.

## Quick Start

```bash
# 1. Setup
uv venv && source .venv/bin/activate
CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir
uv pip install -r requirements.txt

# 2. Configure (.env)
WORKBENCH_API_KEY="your-secret-key"
GEMINI_API_KEY="optional-for-cloud-summarization"
CHAT_MODEL_ENABLED=true
CHAT_MODEL_PATH=models/gpt-oss-20b-F16.gguf

# 3. Download chat model (optional, ~14GB)
huggingface-cli download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-F16.gguf --local-dir models

# 4. Run
uvicorn src.server:app --host localhost --port 8000
```

API docs at `http://localhost:8000/docs`

## OpenAI Agent SDK Compatibility

eGemma implements the **OpenAI Responses API**, making it fully compatible with the OpenAI Agent SDK (`@openai/agents`):

```typescript
import { OpenAI } from "openai";
import { Agent, run } from "@openai/agents";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "your-key",
});

const agent = new Agent({
  name: "local-agent",
  model: "gpt-oss-20b",
  instructions: "You are a helpful assistant.",
  tools: [/* your tools */],
});

const result = await run(agent, "Hello!");
```

### Responses API

The primary endpoint for agentic workflows:

```bash
curl -X POST "http://localhost:8000/v1/responses" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "input": "List files in the current directory",
    "stream": true,
    "tools": [{
      "type": "function",
      "name": "bash",
      "description": "Execute bash command",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {"type": "string"}
        },
        "required": ["command"]
      }
    }]
  }'
```

Supports:

- **Streaming** via Server-Sent Events (SSE)
- **Tool calling** with function definitions
- **Extended thinking** (reasoning traces)
- **Stateful conversations** via `previous_response_id`

### Embeddings

```bash
curl -X POST "http://localhost:8000/embed?dimensions=256" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@document.txt"
```

### Summarization

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@code.py" \
  -F "persona=developer"
```

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/responses` | POST | OpenAI Responses API (Agent SDK compatible) |
| `/v1/conversations` | POST/GET | Conversation management |
| `/v1/conversations/{id}` | GET/DELETE | Get or delete conversation |
| `/v1/conversations/{id}/items` | GET/POST | Conversation items (messages) |
| `/v1/models` | GET | List available models |
| `/embed` | POST | Generate embeddings (128-768 dimensions) |
| `/summarize` | POST | Summarize code/docs with personas |
| `/parse-ast` | POST | Python AST extraction |
| `/health` | GET | Model status |
| `/rate-limits` | GET | Current rate limit config |

See [**Full API Reference →**](docs/api.md) for detailed documentation.

## Configuration

```env
# --- Required ---
WORKBENCH_API_KEY="your-api-key"
HF_TOKEN="..."                # Required for Gemma models (embeddings + summarization)

# --- Chat Model (GPT-OSS) ---
CHAT_MODEL_ENABLED=true
CHAT_MODEL_PATH=models/gpt-oss-20b-F16.gguf
CHAT_N_CTX=65536              # Context window (max 131072)
CHAT_N_GPU_LAYERS=-1          # -1 = all layers on GPU
CHAT_TEMPERATURE=1.0          # Sampling temperature
CHAT_REASONING_EFFORT=medium  # low, medium, high - controls thinking depth

# --- Embeddings (Gemma 300M) ---
FORCE_CPU=false               # Force CPU mode

# --- Summarization (Gemma 12B) ---
SUMMARY_LOCAL_ENABLED=true    # Use local Gemma model

# --- Gemini API (Optional) ---
GEMINI_API_KEY="..."          # Required for Gemini summarization fallback
```

## Extended Thinking

GPT-OSS-20B exposes reasoning traces via Harmony format. When streaming, you'll see:

```text
event: response.reasoning_summary.delta
data: {"delta": "The user wants to list files..."}

event: response.output_text.delta
data: {"delta": "Here are the files..."}
```

Or in responses:

```json
{
  "output": [
    {"type": "reasoning", "summary": [{"text": "Thinking about the request..."}]},
    {"type": "message", "content": [{"text": "Here's my response..."}]}
  ]
}
```

## Personas

Customizable system prompts in `personas/` directory:

- `developer` - Code analysis
- `log_analyst` - Log file processing
- `security_validator` - Threat detection
- See `personas/README.md` for customization

## Hardware Requirements

| Model | RAM/VRAM | Notes |
|-------|----------|-------|
| Embeddings (Gemma 300M) | 1GB | CPU or GPU |
| Summarization (Gemma 12B) | 8GB | GPU recommended |
| Chat (GPT-OSS-20B Q4) | 12-20GB | 32GB RAM (Apple) or 16GB VRAM (CUDA) |

## Notice

**User Control and Responsibility**

- You choose which models to use - none are "officially recommended"
- Personas are starting templates, user-controlled
- No safety guarantees - your deployment, your responsibility

**License:** MIT. NO WARRANTY. NO LIABILITY.
