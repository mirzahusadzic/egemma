# FastAPI API with Gemma and Gemini Models

This project provides a FastAPI application for embedding text using the [Gemma embedding 300m model](https://deepmind.google.com/models/gemma/embeddinggemma) and for summarizing content using either local Gemma models or the Gemini API. It's designed to serve as a robust local server for integrating powerful language model capabilities into your workflow via simple, secure endpoints.

## Key Features

| Feature                      | Description                                                                                                                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dual Model Support**       | Choose between a local [Gemma-based model](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf) for full control or the powerful **Gemini API** for summarization, selectable via an API parameter. |
| **Intelligent Device Detection** | Automatically utilizes available hardware acceleration (CUDA for NVIDIA, MPS for Apple/AMD) with a graceful fallback to CPU for local models.                                                        |
| **Advanced Embeddings**      | Utilizes the Gemma embedding 300m model with Matryoshka support, allowing for variable dimensions (128, 256, 512, 768) to suit different needs.                                                          |
| **Prompt Flexibility**       | Use various prompt types (e.g., 'query', 'document') to generate embeddings optimized for specific tasks, with an optional title for document embeddings.                                               |
| **Content Summarization**    | Generate summaries for code, logs, and Markdown files with support for custom personas (e.g., `developer`, `log_analyst`) to tailor the output.                                                         |
| **Log File Condensation**    | Automatically condenses repetitive log entries before summarization, improving summary quality for verbose log data.                                                                                   |
| **Performance Caching**      | Improves performance by caching embedding results for frequently requested texts using an LRU cache.                                                                                                   |
| **Secure and Robust**        | Includes bearer token authentication to secure endpoints and an in-memory rate limiter to prevent abuse and manage resource usage effectively.                                                         |
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
    # Set to false to disable the /summarize endpoint
    SUMMARY_ENABLED=true
    # Default local model for summarization
    SUMMARY_MODEL_NAME="google/gemma-3-12b-it-qat-q4_0-gguf"
    SUMMARY_MODEL_BASENAME="gemma-3-12b-it-q4_0.gguf"
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

The API includes an in-memory rate limiter to prevent abuse. It operates on a per-client IP basis and is configured separately for the `/embed` and `/summarize` endpoints. If the limit is exceeded, the API will respond with a `429 Too Many Requests` error.

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

| Parameter     | Type      | Description                                                                                                                                                           |
| ------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`  | `string`  | Optional. The model to use (e.g., `gemini-1.5-flash`). If omitted, the default local Gemma model is used.                                                                |
| `persona`     | `string`  | Optional. The persona for summarization (`developer`, `assistant`, `log_analyst`). Defaults based on file type.                                                          |
| `max_tokens`  | `integer` | Optional. Maximum number of tokens for the summary.                                                                                                                   |
| `temperature` | `float`   | Optional. Generation temperature (e.g., `0.2` for deterministic, `0.8` for creative).                                                                                   |

> **Note on `max_tokens` and Persona:** While `max_tokens` in the query parameter sets a hard limit on the generated output length, including `{max_tokens}` within your persona's system message (e.g., "Aim to summarize within {max_tokens} tokens.") can significantly improve the quality and coherence of the summary within that limit. The model uses this internal guidance to better plan and prioritize its output, even if the hard limit is eventually reached.

**Example Request (using Gemini):**

```bash
curl -X POST 'http://localhost:8000/summarize?model_name=gemini-2.5-flash&persona=developer' \
  -H 'Authorization: Bearer your_api_key_here' \
  -F 'file=@/path/to/your/file.py'
```

**Example Request (using local Gemma):**

```bash
curl -X POST 'http://localhost:8000/summarize?persona=developer' \
  -H 'Authorization: Bearer your_api_key_here' \
  -F 'file=@/path/to/your/file.py'
```

**Example Response:**

```json
{
  "language": "Python",
  "summary": "This is a summary of the Python code generated by the specified model."
}
```
