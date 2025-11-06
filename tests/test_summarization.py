from unittest.mock import MagicMock, patch

from src.summarization import SummarizationModelWrapper


def test_summarization_wrapper_init():
    """Test SummarizationModelWrapper initialization."""
    wrapper = SummarizationModelWrapper()
    assert wrapper.model is None
    assert wrapper.gemini_client is None


def test_load_local_model():
    """Test loading local Llama model."""
    wrapper = SummarizationModelWrapper()

    with patch("llama_cpp.Llama.from_pretrained") as mock_from_pretrained:
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        wrapper.load_local_model()

        assert wrapper.model is not None
        mock_from_pretrained.assert_called_once()


def test_load_gemini_client_with_api_key(monkeypatch):
    """Test loading Gemini client when API key is set."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    from src.config import Settings

    settings = Settings()

    wrapper = SummarizationModelWrapper()

    with patch("google.genai.Client") as mock_client:
        with patch("src.summarization.settings", settings):
            wrapper.load_gemini_client()

        assert wrapper.gemini_client is not None
        mock_client.assert_called_once_with(api_key="test-key")


def test_load_gemini_client_without_api_key():
    """Test loading Gemini client when API key is not set."""
    wrapper = SummarizationModelWrapper()

    with patch("src.summarization.settings") as mock_settings:
        mock_settings.GEMINI_API_KEY = None
        wrapper.load_gemini_client()

    assert wrapper.gemini_client is None


def test_summarize_gemini_client_error_429():
    """Test Gemini API quota exceeded error (429)."""
    wrapper = SummarizationModelWrapper()
    wrapper.gemini_client = MagicMock()

    from google.genai import errors

    # Create a real ClientError instance
    mock_error = errors.ClientError(code=429, response_json={})

    wrapper.gemini_client.models.generate_content.side_effect = mock_error

    result = wrapper._summarize_gemini(
        content="test content",
        language="Python",
        persona_name="developer",
        max_tokens=100,
        temperature=0.2,
        model_name="gemini-2.5-flash",
        enable_safety=False,
    )

    assert "quota exceeded" in result.lower()
    assert "try again later" in result.lower()


def test_summarize_gemini_client_error_503():
    """Test Gemini API service unavailable error (503)."""
    wrapper = SummarizationModelWrapper()
    wrapper.gemini_client = MagicMock()

    from google.genai import errors

    mock_error = errors.ClientError(code=503, response_json={})

    wrapper.gemini_client.models.generate_content.side_effect = mock_error

    result = wrapper._summarize_gemini(
        content="test content",
        language="Python",
        persona_name="developer",
        max_tokens=100,
        temperature=0.2,
        model_name="gemini-2.5-flash",
        enable_safety=False,
    )

    assert "overloaded" in result.lower()
    assert "try again" in result.lower()


def test_summarize_gemini_client_error_other():
    """Test other Gemini API client errors."""
    wrapper = SummarizationModelWrapper()
    wrapper.gemini_client = MagicMock()

    from google.genai import errors

    mock_error = errors.ClientError(code=400, response_json={})

    wrapper.gemini_client.models.generate_content.side_effect = mock_error

    result = wrapper._summarize_gemini(
        content="test content",
        language="Python",
        persona_name="developer",
        max_tokens=100,
        temperature=0.2,
        model_name="gemini-2.5-flash",
        enable_safety=False,
    )

    assert "API error" in result
    assert "400" in result


def test_summarize_gemini_unexpected_error():
    """Test unexpected error during Gemini API call."""
    wrapper = SummarizationModelWrapper()
    wrapper.gemini_client = MagicMock()

    wrapper.gemini_client.models.generate_content.side_effect = RuntimeError(
        "Unexpected error"
    )

    result = wrapper._summarize_gemini(
        content="test content",
        language="Python",
        persona_name="developer",
        max_tokens=100,
        temperature=0.2,
        model_name="gemini-2.5-flash",
        enable_safety=False,
    )

    assert "unexpected error" in result.lower()


def test_summarize_gemini_no_candidates():
    """Test when Gemini returns no candidates."""
    wrapper = SummarizationModelWrapper()
    wrapper.gemini_client = MagicMock()

    mock_response = MagicMock()
    mock_response.candidates = []

    wrapper.gemini_client.models.generate_content.return_value = mock_response

    result = wrapper._summarize_gemini(
        content="test content",
        language="Python",
        persona_name="developer",
        max_tokens=100,
        temperature=0.2,
        model_name="gemini-2.5-flash",
        enable_safety=False,
    )

    assert "no candidates" in result.lower()


def test_summarize_gemini_no_content_parts():
    """Test when Gemini returns no content parts."""
    wrapper = SummarizationModelWrapper()
    wrapper.gemini_client = MagicMock()

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.content.parts = []
    mock_response.candidates = [mock_candidate]

    wrapper.gemini_client.models.generate_content.return_value = mock_response

    result = wrapper._summarize_gemini(
        content="test content",
        language="Python",
        persona_name="developer",
        max_tokens=100,
        temperature=0.2,
        model_name="gemini-2.5-flash",
        enable_safety=False,
    )

    assert "no content parts" in result.lower()
