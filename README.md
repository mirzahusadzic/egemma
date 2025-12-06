# FastAPI API with Gemma and Gemini Models

<div align="center" style="margin-top: 20px; margin-bottom: 20px;">
<img src="./docs/assets/egemma-logo.png" alt="eGemma Logo" width="512"/>
</div>

A high-performance FastAPI workbench that brings the power of Google's Gemma and Gemini models to your local machine. This application provides two core services through a secure and easy-to-use API:

* **Advanced Text Embeddings:** Generate sophisticated text embeddings using the [Gemma embedding 300m model](https://deepmind.google.com/models/gemma/embeddinggemma), featuring Matryoshka support for variable dimensions (128-768) to tailor embeddings for diverse tasks.

* **Intelligent Content Summarization:** Summarize code, logs, and documents using either a local [Gemma-based model](https://deepmind.google/models/gemma/gemma-3) for full control and privacy, or the powerful [Gemini API for text generation](https://ai.google.dev/gemini-api/docs/text-generation) for cutting-edge performance. Summaries can be tailored with different personas (`developer`, `log_analyst`, etc.) for context-aware results.

Designed for robust local deployment, the server automatically detects and utilizes available hardware acceleration (NVIDIA CUDA, Apple/AMD MPS) and includes features like performance caching, rate limiting, and secure bearer token authentication. It's a powerful tool for integrating advanced language model capabilities into your local development workflow.

## Important Notice

⚠️ **User Control and Responsibility**

* **You choose which models to use**: Gemma (local) or Gemini (API) - neither is "officially recommended" or "more secure"
* **Personas are provisional**: The included personas (`personas/`) are starting templates, user-controlled, and NOT officially endorsed
* **No safety guarantees**: This workbench does NOT provide content filtering, safety checks, or ethical guardrails
* **Your deployment, your responsibility**: How you configure and use this tool is entirely up to you

See `personas/README.md` for details on persona usage and customization.

**License**: MIT. NO WARRANTY. NO LIABILITY.

## Key Features

Here are some of the standout features of this application:

| Feature                      | Description                                                                                                                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Multi-Model Support**      | Run up to three models simultaneously: a local Gemma for embeddings, and your choice of a local Gemma or the Gemini API for summarization. |
| **Intelligent Device Detection** | Automatically utilizes available hardware acceleration (CUDA for NVIDIA, MPS for Apple/AMD) with a graceful fallback to CPU for local models.                                                        |
| **Advanced Embeddings**      | Utilizes the Gemma embedding 300m model with Matryoshka support, allowing for variable dimensions (128, 256, 512, 768) to suit different needs.                                                          |
| **Prompt Flexibility**       | Use various prompt types (e.g., 'query', 'document') to generate embeddings optimized for specific tasks, with an optional title for document embeddings.                                               |
| **Content Summarization**    | Generate summaries for code, logs, and Markdown files with support for custom personas (e.g., `developer`, `log_analyst`) to tailor the output.                                                         |
| **Log File Condensation**    | Automatically condenses repetitive log entries before summarization, improving summary quality for verbose log data.                                                                                   |
| **Performance Caching**      | Improves performance by caching embedding and summary results for frequently requested texts using an LRU cache.                                                                                                   |
| **Secure and Robust**        | Includes bearer token authentication to secure endpoints and an in-memory rate limiter to prevent abuse, manage resource usage, and provide crucial cost protection when using paid APIs like Gemini. |
| **Modern API Interface**     | A high-performance API built with FastAPI, including automatic interactive documentation with a sleek dark theme.                                                                                        |

## Setup

Follow these steps to set up the project locally:

1. **Create Virtual Environment:**
    It is recommended to use `uv` for managing your Python environment.

    ```bash
    uv venv
    source .venv/bin/activate
    ```

2. **Install Dependencies:**
    > **Note on `llama-cpp-python`:** For specific hardware acceleration (e.g., Metal for Apple Silicon), you may need to install `llama-cpp-python` with special flags *before* installing other requirements.

    ```bash
    CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir
    ```

    **Install** the required Python packages:

    ```bash
    uv pip install -r requirements.txt
    ```

    > **Note on Gemini SDK:** This project now uses the unified `google-genai` SDK (v0.3.0+) which provides full support for Gemini 2.5 models. The legacy `google-generativeai` SDK is deprecated and will reach end-of-life on August 31, 2025.

3. **Environment Variables:**
    Create a `.env` file in the project root. Add the following configurations:

    ```env
    # --- Required ---
    # For authenticating API requests
    WORKBENCH_API_KEY="your_api_key_here"

    # --- Gemini API (Optional) ---
    # Required if you want to use Gemini for summarization
    GEMINI_API_KEY="your_google_ai_studio_api_key"

    # --- Hugging Face (Optional) ---
    # For private models or to avoid rate limits
    HF_TOKEN="your_huggingface_token_here"

    # --- Summarization Settings ---
    # Set to false to disable local summarization (Gemini API will still work)
    SUMMARY_LOCAL_ENABLED=true
    # Default local model for summarization
    SUMMARY_MODEL_NAME="google/gemma-3-12b-it-qat-q4_0-gguf"
    SUMMARY_MODEL_BASENAME="gemma-3-12b-it-q4_0.gguf"

    # --- Hardware Acceleration (Optional) ---
    # Force CPU usage for embeddings (useful for GPU compatibility issues)
    FORCE_CPU=false
    ```

* Replace placeholder values with your actual keys.
* You can obtain a `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Running the Server

To start the FastAPI server, run:

```bash
uvicorn src.server:app --host localhost --port 8000
```

The interactive API documentation will be available at `http://localhost:8000/docs`.

## API Usage

### Rate Limiting

The API includes an in-memory rate limiter to prevent abuse and help manage costs when using paid services like the Gemini API. It operates on a per-client IP basis and is configured separately for the `/embed` and `/summarize` endpoints. If the limit is exceeded, the API will respond with a `429 Too Many Requests` error.

### Endpoints

#### Embed File Content

* **Endpoint:** `POST /embed`
* **Authentication:** Bearer Token (`WORKBENCH_API_KEY`)
* **Description:** Embeds file content using the Gemma 300m model with Matryoshka support.

**Query Parameters:**

| Parameter     | Type      | Description                                                                                             |
| ------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| `dimensions`  | `integer` | Optional. Embedding dimensions. Valid values: `128`, `256`, `512`, `768`. Defaults to `128`.              |
| `prompt_name` | `string`  | Optional. Prompt type to optimize embeddings (`query`, `document`, `clustering`). Defaults to `document`. |
| `title`       | `string`  | Optional. A title for the content, useful when `prompt_name` is `document`.                               |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/embed?dimensions=128" \
     -H "Authorization: Bearer your_api_key_here" \
     -F "file=@/path/to/your/document.txt"
```

#### Summarize Content

* **Endpoint:** `POST /summarize`
* **Authentication:** Bearer Token (`WORKBENCH_API_KEY`)
* **Description:** Summarizes a code, Markdown, or log file using either a local Gemma model or the Gemini API.

**Query Parameters:**

| Parameter       | Type      | Description                                                                                                                                                           |
| --------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`    | `string`  | Optional. The model to use (e.g., `gemini-2.5-flash`). If omitted, the default local Gemma model is used.                                                                |
| `persona`       | `string`  | Optional. The persona for summarization (`developer`, `assistant`, `log_analyst`, `security_validator`). Defaults based on file type.                                     |
| `max_tokens`    | `integer` | Optional. Maximum number of tokens for the summary.                                                                                                                   |
| `temperature`   | `float`   | Optional. Generation temperature (e.g., `0.2` for deterministic, `0.8` for creative).                                                                                   |
| `enable_safety` | `boolean` | Optional. Enable Gemini safety settings for content filtering (only applies to Gemini models). Default: `false`.                                                         |

> **Note on `max_tokens` and Persona:** While `max_tokens` in the query parameter sets a hard limit on the generated output length, including `{max_tokens}` within your persona's system message (e.g., "Aim to summarize within {max_tokens} tokens.") can significantly improve the quality and coherence of the summary within that limit. The model uses this internal guidance to better plan and prioritize its output, even if the hard limit is eventually reached.

**Example Request (using Gemini):**

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer your_api_key_here" \
  -F "file=@/path/to/your/file.py" \
  -F "model_name=gemini-2.5-flash" \
  -F "persona=developer"
```

**Example Request (using local Gemma):**

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer your_api_key_here" \
  -F "file=@/path/to/your/file.py" \
  -F "persona=developer"
```

**Example Request (security validation with Gemini):**

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Authorization: Bearer your_api_key_here" \
  -F "file=@/path/to/VISION.md" \
  -F "model_name=gemini-2.5-flash" \
  -F "persona=security_validator" \
  -F "enable_safety=true"
```

**Example Response:**

```json
{
  "language": "Markdown",
  "summary": "THREAT ASSESSMENT: SAFE\n\nDETECTED PATTERNS: None\n\nSPECIFIC CONCERNS: None\n\nRECOMMENDATION: APPROVE\n\nREASONING: Document contains clear security principles with no manipulative patterns detected."
}
```

#### Parse AST

* **Endpoint:** `POST /parse-ast`
* **Authentication:** Bearer Token (`WORKBENCH_API_KEY`)
* **Description:** Parses a code file (currently Python only) and returns its Abstract Syntax Tree (AST) representation, including structural information like imports, classes, and functions with detailed metadata.

**Form Data Parameters:**

| Parameter  | Type     | Description                                                                                                                                                                                                                                                                                                                                                                   |
| ---------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `file`     | `file`   | The code file to parse.                                                                                                                                                                                                                                                                                                                                                       |
| `language` | `string` | The programming language of the file (e.g., `python`).                                                                                                                                                                                                                                                                                                                        |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/parse-ast" \
     -H "Authorization: Bearer your_api_key_here" \
     -F "file=@/path/to/your/code.py" \
     -F "language=python"
```

**Example Response (Python):**

```json
{
  "language": "python",
  "docstring": "A test module for parsing.",
  "imports": [
    "os",
    "typing"
  ],
  "functions": [
    {
      "name": "my_function",
      "docstring": "A standalone function.",
      "params": [
        {
          "name": "param1",
          "type": "None"
        },
        {
          "name": "param2",
          "type": "None"
        }
      ],
      "returns": "int",
      "decorators": [],
      "is_async": false
    }
  ],
  "classes": [
    {
      "name": "MyClass",
      "docstring": "A simple example class.",
      "base_classes": [],
      "methods": [
        {
          "name": "__init__",
          "docstring": "",
          "params": [
            {
              "name": "self",
              "type": "None"
            }
          ],
          "returns": "None",
          "decorators": [],
          "is_async": false
        },
        {
          "name": "my_method",
          "docstring": "",
          "params": [
            {
              "name": "self",
              "type": "None"
            },
            {
              "name": "arg1",
              "type": "int"
            }
          ],
          "returns": "int",
          "decorators": [],
          "is_async": false
        }
      ],
      "decorators": []
    }
  ]
}
```

#### Health Check

* **Endpoint:** `GET /health`
* **Authentication:** None
* **Description:** Provides the current status of the API and its loaded models.

**Example Request:**

```bash
curl -X GET "http://localhost:8000/health"
```

**Example Response:**

```json
{
  "status": "ok",
  "embedding_model": {
    "name": "google/embeddinggemma-300m",
    "status": "loaded"
  },
  "local_summarization_model": {
    "name": "stduhpf/google-gemma-3-12b-it-qat-q4_0-gguf-small",
    "basename": "gemma-3-12b-it-q4_0_s.gguf",
    "status": "loaded"
  },
  "gemini_api": {
    "api_key_set": true,
    "default_model": "gemini-2.5-flash"
  }
}
```

#### Rate Limits

* **Endpoint:** `GET /rate-limits`
* **Authentication:** None
* **Description:** Returns the current rate limit configuration. Clients can query this endpoint on startup to configure their rate limiters adaptively.

**Example Request:**

```bash
curl -X GET "http://localhost:8000/rate-limits"
```

**Example Response:**

```json
{
  "embed": {
    "calls": 5,
    "seconds": 10
  },
  "summarize": {
    "calls": 2,
    "seconds": 60
  }
}
```

This allows clients to respect the server's actual rate limits rather than relying on hardcoded values.

### Personas

Personas are system prompts that guide the model's behavior for specific use cases. They are stored as Markdown files in the `personas/` directory:

* **`personas/code/`** - For code summarization
  * `developer.md` - General code understanding
  * `log_analyst.md` - Log file analysis
  * `structural_analyst.md` - Code structure analysis
  * `sdet.md` - Test-focused analysis

* **`personas/docs/`** - For document analysis
  * `assistant.md` - General documentation
  * `security_validator.md` - Security threat detection

#### Security Validator Persona

The `security_validator` persona is designed for detecting malicious patterns in strategic documents. It analyzes content for 5 attack vectors:

1. **Security Weakening** - Phrases suggesting reduced validation or bypassed checks
2. **Trust Erosion** - Language undermining proof-based systems
3. **Permission Creep** - Gradual expansion of access rights
4. **Ambiguity Injection** - Vague language exploitable for malicious interpretation
5. **Velocity Over Safety** - Speed emphasis suggesting skipped security measures

**Response Format:**

```text
THREAT ASSESSMENT: [SAFE | SUSPICIOUS | MALICIOUS]
DETECTED PATTERNS: [List of patterns found]
SPECIFIC CONCERNS: [Quoted suspicious phrases with context]
RECOMMENDATION: [APPROVE | REVIEW | REJECT]
REASONING: [Brief explanation]
```

**Use with `enable_safety=true`** for Gemini's built-in content filtering alongside persona-based analysis.
