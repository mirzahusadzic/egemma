# eGemma

<div align="center">
<img src="./docs/assets/egemma-logo.png" alt="eGemma" width="320"/>

Local inference for embeddings, semantic parsing, chat, and summarization.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/mirzahusadzic/egemma/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mirzahusadzic/egemma/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

| Capability | Model | Interface |
| --- | --- | --- |
| Embeddings | EmbeddingGemma 300M | File and text-batch endpoints; 128, 256, 512, or 768 dimensions |
| Semantic parsing | AMRBART | Attributed AMR graphs, typed triples, and diagnostic results |
| Chat | GPT-OSS via llama.cpp | Responses-style API, SSE streaming, and function calls |
| Summarization | Gemma 3 12B or Gemini API | File summaries with configurable [personas](personas/README.md) |

Bearer authentication and in-memory rate limits protect inference endpoints.
Models run locally; Gemini summarization is an optional cloud service.

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
```

`llama-cpp-python` may require a C/C++ toolchain. For accelerated chat and local
summarization, follow its [Metal/CUDA build instructions](https://github.com/abetlen/llama-cpp-python#installation).
EmbeddingGemma and AMRBART use PyTorch's available CPU, CUDA, or MPS backend.

Accept the access terms on the [EmbeddingGemma model page](https://huggingface.co/google/embeddinggemma-300m),
then authenticate for model downloads:

```bash
uv run hf auth login
```

Create a `.env` file. This example starts the semantic profile:

```env
WORKBENCH_API_KEY=replace-with-a-long-random-key
MODEL_PROFILE=semantic
```

```bash
uv run uvicorn src.server:app --host localhost --port 8000
```

The first start downloads the configured embedding and AMR checkpoints.
Open [API docs](http://localhost:8000/docs); `/health` reports loaded models and revisions.

| Profile | Startup behavior |
| --- | --- |
| `semantic` | Loads EmbeddingGemma and AMRBART; disables chat, summarization, and conversation storage |
| `workbench` (default) | Loads EmbeddingGemma and the enabled chat/summarization models; AMRBART is opt-in |

For chat, download a compatible [GPT-OSS GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)
and configure its actual path. Set these values in `.env`:

```env
MODEL_PROFILE=workbench
CHAT_MODEL_ENABLED=true
CHAT_MODEL_PATH=models/your-gpt-oss-model.gguf
SUMMARY_LOCAL_ENABLED=false
AMRBART_MODEL_ENABLED=false
```

Set `CHAT_MODEL_ENABLED=false` for a workbench without chat. Local Gemma
summarization requires its model access terms and `SUMMARY_LOCAL_ENABLED=true`;
`GEMINI_API_KEY` enables the cloud alternative in the workbench profile.
Memory requirements depend on model precision, context size, and backend.

## Semantic API

Export `WORKBENCH_API_KEY` in your client shell with the same value as the server's
`.env`. Values in `.env` are server configuration, not shell exports; Hugging Face
authentication uses `hf auth login` or an exported `HF_TOKEN`.

```bash
curl -sS http://localhost:8000/amr/parse \
  -H "Authorization: Bearer $WORKBENCH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Can we adopt the parser library?"}'

curl -sS http://localhost:8000/embed/texts \
  -H "Authorization: Bearer $WORKBENCH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"texts":["candidate relation","canonical relation"],"dimensions":768,"prompt_name":"STS"}'
```

AMR results retain the raw decoded text in `linearized_raw`, an indented display
in `linearized`, and an AMR-aware PENMAN graph with matching typed `triples`.
These are linguistic interpretations; downstream systems decide their evidential
and operational use.

`POST /amr/parse-batch` accepts `{"texts":["First sentence.","Second sentence."]}`
and returns one result per input, in order. Check each `decode_status`:

- `ok`: one complete graph was decoded.
- `invalid`: raw model output and diagnostic fragments are retained; `penman` is `null`.
- `input_too_long`: inference was skipped; no graph was generated.

The default AMR limit is **400 source tokens including boundary tokens**. Inputs
are never silently truncated. See the [API reference](docs/api.md) for batch,
character, and output bounds. EmbeddingGemma has a separate 2,048-token window;
split long embedding inputs to avoid encoder truncation.

## OpenAI Agents SDK

The local `/v1/responses` endpoint implements a subset of the Responses API for
text, function calls, and streaming. Hosted OpenAI tools and full API parity are
not provided. The examples in [`scripts/`](scripts/) use `@openai/agents` 0.3.x;
verify your workflow when changing SDK versions.

```typescript
import { OpenAI } from "openai";
import { Agent, run, setDefaultOpenAIClient, setTracingDisabled } from "@openai/agents";

setDefaultOpenAIClient(new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: process.env.WORKBENCH_API_KEY,
}));
setTracingDisabled(true); // Disable hosted trace export.

const agent = new Agent({
  name: "local-agent",
  model: "gpt-oss-20b",
  instructions: "You are a helpful assistant.",
});

const result = await run(agent, "Hello!");
console.log(result.finalOutput);
```

Tool definitions describe calls for the client to execute. The server does not
execute shell commands or other supplied functions.

## Configuration and development

Settings are defined in [`src/config.py`](src/config.py) and read from `.env` or
environment variables. Restart the server after changing them.

| Setting | Default |
| --- | --- |
| `CHAT_N_CTX` | `65536` |
| `CHAT_TEMPERATURE` | `0.3` |
| `CHAT_REASONING_EFFORT` | `high` |
| `AMRBART_MAX_SOURCE_TOKENS` | `400`, including text and AMR boundary tokens |
| `AMRBART_MAX_GRAPH_TOKENS` | `1024` |
| `AMRBART_BATCH_MAX_INPUTS` | `64` |
| `EMBEDDING_BATCH_MAX_INPUTS` | `256` |

Rate limits are per client IP, HTTP method, and route template, within one server
process. `/rate-limits` reports the configured budgets. Run a single worker for
shared local model state and these limits.

```bash
uv run ruff check .
uv run pytest
```

Tests use mocked inference and dummy credentials; model downloads are disabled.
The [API reference](docs/api.md) covers embeddings, AMR, responses, conversations,
summarization, Python AST extraction, and monitoring.

## License

eGemma's original code is [MIT-licensed](LICENSE). The bundled GPT-OSS reference
template is Apache-2.0; dependencies and model weights retain their own licenses
and terms. See [third-party notices](THIRD_PARTY_NOTICES.md), including PENMAN,
AMRBART, and the Gemma models. Model weights are not included in this repository.
