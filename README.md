# FastAPI Embedding API with Gemma Model

This project provides a FastAPI application for embedding text using the [Gemma embedding 300m model](https://deepmind.google/models/gemma/embeddinggemma/) with Matryoshka support. It's designed to serve as a robust local server for integrating Gemma embeddings into your workflow via a simple endpoint.

## Key Features

*   **Intelligent Device Detection:** Automatically utilizes available hardware acceleration (CUDA for NVIDIA GPUs, MPS for Apple Silicon/AMD GPUs) with a graceful fallback to CPU.
*   **Local Gemma Embedding:** Utilize the powerful Gemma embedding 300m model directly on your local machine.
*   **Sentence Caching:** Improves performance by caching embedding results for frequently requested texts using `lru_cache`.
*   **Bearer Token Authentication:** Secure your embedding endpoint with a configurable API key, ensuring only authorized access.
*   **Matryoshka Support:** Generate embeddings with various dimensions (128, 256, 512, 768) to suit different application needs.
*   **FastAPI Interface:** A user-friendly and high-performance API built with FastAPI, including automatic interactive documentation.

## Setup

Follow these steps to set up the project locally:

1.  **Create and Activate Virtual Environment:**
    It is recommended to use `uv` for managing your Python environment.
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    Install the required Python packages:
    ```bash
    uv pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Create a `.env` file in the root of the project. This file should contain your `WORKBENCH_API_KEY` and optionally your `HF_TOKEN` for Hugging Face models.

    ```
    WORKBENCH_API_KEY="your_api_key_here"
    HF_TOKEN="your_huggingface_token_here" # Optional, for private models or rate limits (required for Gemma)
    ```
    Replace `"your_api_key_here"` with your actual API key. This key will be used as a bearer token for authenticating requests to the `/embed` endpoint. If you are using private Hugging Face models or encountering rate limits, replace `"your_huggingface_token_here"` with your Hugging Face API token. Alternatively, you can log in via the Hugging Face CLI: `huggingface-cli login`.

## Running the Server

To start the FastAPI server, run the following command:

```bash
uvicorn src.server:app --host localhost --port 8000
```

The API documentation will be available at `http://localhost:8000/docs`, featuring a sleek dark theme.

## API Usage

### Embed Text

**Endpoint:** `POST /embed`
**Authentication:** Bearer Token (using `WORKBENCH_API_KEY`)
**Description:** Embeds a given text using the Gemma embedding 300m model with Matryoshka support.

**Request Body:**

```json
{
  "text": "Your input text here."
}
```

**Query Parameters:**

*   `dimensions`: Optional. A list of desired embedding dimensions. Valid values are `128`, `256`, `512`, `768`. If not provided, all dimensions will be returned.

**Example Request (using curl):**

```bash
curl -X POST "http://localhost:8000/embed?dimensions=128&dimensions=256" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key_here" \
     -d '{
       "text": "Hello, world!"
     }'
```

**Example Response:**

```json
{
  "embedding_128d": [...],
  "embedding_256d": [...]
}
```