"""Stable test-process configuration independent of a developer's local `.env`."""

import os

# Tests opt into AMRBART explicitly; runtime `.env` choices must not alter that
# baseline.
os.environ["AMRBART_MODEL_ENABLED"] = "false"
os.environ["SUMMARY_MODEL_ENABLED"] = "true"
os.environ["WORKBENCH_API_KEY"] = "test-key"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
