"""Stable test-process configuration independent of a developer's local `.env`."""

import os

# The legacy suite exercises the complete workbench unless a test explicitly
# selects another profile. Runtime `.env` choices must not alter that baseline.
os.environ["MODEL_PROFILE"] = "workbench"
os.environ["AMRBART_MODEL_ENABLED"] = "false"
os.environ["CHAT_MODEL_ENABLED"] = "true"
os.environ["SUMMARY_LOCAL_ENABLED"] = "true"
os.environ["WORKBENCH_API_KEY"] = "test-key"
os.environ["GEMINI_API_KEY"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
