# FastAPI API with Gemma Models for Embedding and Summarization

This project provides a FastAPI application for embedding text using the [Gemma embedding 300m model](https://deepmind.google/models/gemma/embeddinggemma) with Matryoshka support, and for summarizing code and Markdown files using a [Gemma-based summarization model](https://deepmind.google/models/gemma/gemma-3). It's designed to serve as a robust local server for integrating Gemma capabilities into your workflow via simple endpoints.

## Key Features

*   **Prompt Flexibility for Embeddings:** Utilize various prompt types (e.g., 'query', 'document', 'clustering') to generate embeddings optimized for specific tasks, with an optional title for document embeddings.
*   **Intelligent Device Detection:** Automatically utilizes available hardware acceleration (CUDA for NVIDIA GPUs, MPS for Apple Silicon/AMD GPUs) with a graceful fallback to CPU.
* **Local Gemma Embedding:** Utilize the powerful Gemma embedding 300m model directly on your local machine.
* **Log File Condensation:** Automatically condenses repetitive log entries before summarization, improving the quality of summaries for verbose log data.
* **Code Summarization:** Generate summaries for code and Markdown files with an optional, configurable endpoint.
* **Sentence Caching:** Improves performance by caching embedding results for frequently requested texts using `lru_cache`.
* **Bearer Token Authentication:** Secure your embedding endpoint with a configurable API key, ensuring only authorized access.
* **Matryoshka Support:** Generate embeddings with various dimensions (128, 256, 512, 768) to suit different application needs.
* **FastAPI Interface:** A user-friendly and high-performance API built with FastAPI, including automatic interactive documentation.

## Setup

Follow these steps to set up the project locally:

1. **Create (and optionally activate) Virtual Environment:**
    It is recommended to use `uv` for managing your Python environment.

    ```bash
    uv venv
    source .venv/bin/activate
    ```

2. **Install Dependencies:**

    **Note on `llama-cpp-python` installation:**

    >If you intend to use Metal (for Apple Silicon/AMD GPUs) or other specific hardware acceleration
    for `llama-cpp-python`, you might need to install it separately with specific build flags
    *before* running `uv pip install -r requirements.txt`. For example, for Metal support:

    ```bash
    CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir
    ```

    Install the required Python packages:

    ```bash
    uv pip install -r requirements.txt
    ```

3. **Environment Variables:**
    Create a `.env` file in the root of the project. This file should contain your `WORKBENCH_API_KEY` and other optional settings.

    ```bash
    # --- Required --- 
    # For authenticating API requests
    WORKBENCH_API_KEY="your_api_key_here"

    # --- Optional --- 
    # For private models or to avoid rate limits (required for Gemma)
    HF_TOKEN="your_huggingface_token_here"

    # --- Summarization Settings --- 
    # Set to false to disable the /summarize endpoint
    SUMMARY_ENABLED=true
    # Hugging Face repository ID for the summarization model (default: google/gemma-3-12b-it-qat-q4_0-gguf)
    SUMMARY_MODEL_NAME="google/gemma-3-12b-it-qat-q4_0-gguf"
    # Filename of the GGUF model within the repository (default: gemma-3-12b-it-q4_0.gguf)
    SUMMARY_MODEL_BASENAME="gemma-3-12b-it-q4_0.gguf"
    # Maximum number of tokens for the summary (default: 300)
    SUMMARY_MAX_TOKEN=1024
    # Temperature for the summary generation (default: 0.2)
    SUMMARY_TEMP=0.2
    ```

    Replace `"your_api_key_here"` with your actual API key. This key will be used as a bearer token for authenticating requests to the `/embed` endpoint. If you are using private Hugging Face models or encountering rate limits, replace `"your_huggingface_token_here"` with your Hugging Face API token. Alternatively, you can log in via the Hugging Face CLI: `huggingface-cli login`.

## Running the Server

To start the FastAPI server, run the following command:

```bash
uv run uvicorn src.server:app --host localhost --port 8000
```

The API documentation will be available at `http://localhost:8000/docs`, featuring a sleek dark theme.

## API Usage

### Embed File Content

**Endpoint:** `POST /embed`
**Authentication:** Bearer Token (using `WORKBENCH_API_KEY`)
**Description:** Embeds the content of an uploaded file using the Gemma embedding 300m model with Matryoshka support.

**Request Body:**

This endpoint uses a `multipart/form-data` request to handle file uploads. The file should be sent under the `file` key.

**Query Parameters:**

*   `dimensions`: Optional. A list of desired embedding dimensions. Valid values are `128`, `256`, `512`, `768`. If not provided, the `embedding_128d` will be returned.
*   `prompt_name`: Optional. The name of the prompt to use for the embedding model. This helps optimize the embedding for specific tasks (e.g., `query`, `document`, `clustering`). Defaults to `document`.
*   `title`: Optional. A title for the uploaded content. This is particularly useful when `prompt_name` is set to `document`, as it helps the model generate more relevant embeddings by incorporating the document's title.

**Example Request (using curl):**

```bash
curl -X POST "http://localhost:8000/embed?dimensions=128&prompt_name=document&title=My%20Document%20Title" \
     -H "Authorization: Bearer your_api_key_here" \
     -F 'file=@/path/to/your/document.txt'
```

**Example Response:**

```json
{
  "embedding_128d": [...]
}
```

### Summarize Code, Markdown, or Log File

**Endpoint:** `POST /summarize`
**Authentication:** Bearer Token (using `WORKBENCH_API_KEY`)
**Description:** Upload a code, Markdown, or log file and get a summary. Log files are automatically condensed before summarization. Summarization is guided by a configurable persona.

**Request Body:**

This endpoint uses a `multipart/form-data` request to handle file uploads. The file should be sent under the `file` key.

**Query Parameters:**

* `max_tokens`: Optional. Maximum number of tokens for the summary. (e.g., `?max_tokens=512`)
* `temperature`: Optional. Temperature for the summary generation. Lower values (e.g., `0.2`) make the output more deterministic, higher values (e.g., `0.8`) make it more creative. (e.g., `&temperature=0.7`)
* `persona`: Optional. The name of the persona to use for summarization. This corresponds to a Markdown file in the `personas/code` or `personas/docs` directory (e.g., `developer`, `assistant`, `log_analyzer`). If not provided, a default persona will be used based on the file type.

    **Note on `max_tokens` and Persona:** While `max_tokens` in the query parameter sets a hard limit on the generated output length, including `{max_tokens}` within your persona's system message (e.g., "Aim to summarize within {max_tokens} tokens.") can significantly improve the quality and coherence of the summary within that limit. The model uses this internal guidance to better plan and prioritize its output, even if the hard limit is eventually reached.

**Example Request (using curl):**

```bash
curl -X POST 'http://localhost:8000/summarize?max_tokens=1024&temperature=0.2&persona=developer' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your_api_key_here' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/file.py'
```

**Example Response:**

```json
{
  "language": "Python",
  "summary": "This is a summary of the Python code."
}
```
