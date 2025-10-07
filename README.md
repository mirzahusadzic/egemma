# FastAPI Embedding API with Gemma Model

This project provides a FastAPI application for embedding text using the [Gemma embedding 300m model](https://deepmind.google/models/gemma/embeddinggemma/) with Matryoshka support.

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
    Replace `"your_api_key_here"` with your actual API key. If you are using private Hugging Face models or encountering rate limits, replace `"your_huggingface_token_here"` with your Hugging Face API token. Alternatively, you can log in via the Hugging Face CLI: `huggingface-cli login`.

## Running the Server

To start the FastAPI server, run the following command:

```bash
uvicorn server:app --host localhost --port 8002
```

The API documentation will be available at `http://localhost:8002/docs`.
