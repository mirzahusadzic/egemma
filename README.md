# eGemma - Local AI Workbench

<div align="center">
<img src="./docs/assets/egemma-logo.png" alt="eGemma Logo" width="400"/>

A unified API server for embeddings, summarization, and chat completions.
Run powerful AI models locally with optional cloud fallback.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Features

| Capability | Model | Description |
|------------|-------|-------------|
| **Embeddings** | Gemma 300M | Matryoshka support (128-768 dims), optimized prompts |
| **Summarization** | Gemma 12B / Gemini API | Code, logs, docs with persona-based analysis |
| **Chat** | GPT-OSS-20B | OpenAI-compatible API, 128K context, tool calling |

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
CHAT_MODEL_PATH=models/gpt-oss-20b-Q4_K_M.gguf

# 3. Download chat model (optional, ~12GB)
huggingface-cli download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf --local-dir models

# 4. Run
uvicorn src.server:app --host localhost --port 8000
```

API docs at `http://localhost:8000/docs`

## API Overview

### Chat Completions (OpenAI-compatible)

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Works with OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")
response = client.chat.completions.create(model="gpt-oss-20b", messages=[...])
```

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
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming, tools, sessions) |
| `/v1/sessions` | POST/GET | Session management for multi-turn conversations |
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
CHAT_MODEL_PATH=models/gpt-oss-20b-Q4_K_M.gguf
CHAT_N_CTX=65536              # Context window (max 131072)
CHAT_N_GPU_LAYERS=-1          # -1 = all layers on GPU

# --- Embeddings (Gemma 300M) ---
FORCE_CPU=false               # Force CPU mode

# --- Summarization (Gemma 12B) ---
SUMMARY_LOCAL_ENABLED=true    # Use local Gemma model

# --- Gemini API (Optional) ---
GEMINI_API_KEY="..."          # Required for Gemini summarization fallback
```

## Extended Features

### Session Management

Persistent server-side conversation history:

```bash
# Create session
curl -X POST "http://localhost:8000/v1/sessions" -H "Authorization: Bearer $API_KEY"
# Returns: {"session_id": "sess_abc123", ...}

# Use in chat (history auto-prepended)
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "X-Session-Id: sess_abc123" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"messages": [{"role": "user", "content": "Remember my name is Alice"}]}'
```

### Extended Thinking

GPT-OSS-20B exposes reasoning traces via Harmony format:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"messages": [...], "include_thinking": true}'
```

Response includes `thinking` field with model's internal reasoning.

### Personas

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
