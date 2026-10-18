import logging
from unittest.mock import MagicMock, patch

import numpy as np
import penman
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from penman.models.amr import model as amr_model

from src.config import settings
from src.models.amrbart import (
    AMRBartModelWrapper,
    AMRDecodeError,
    AMRFragment,
    AMRParseResult,
    AMRTriple,
    pointer_graph_to_penman,
    scan_invalid_amr_fragments,
)
from src.routers import amr, embed
from src.server import app, get_api_key
from src.util.rate_limiter import _client_last_request_time

RAW_GRAPH = """<AMR> (
  <pointer:0> contrast-01
  :ARG1 (<pointer:1> pass-07 :ARG1 (<pointer:2> library))
  :ARG2 (<pointer:3> change-01 :ARG0 <pointer:2>
                              :ARG1 (<pointer:4> graph))
)</AMR>"""


@pytest.fixture(autouse=True)
def clear_rate_limits():
    _client_last_request_time.clear()
    yield
    _client_last_request_time.clear()


@pytest.fixture
def parser():
    wrapper = AMRBartModelWrapper()
    wrapper.device = "cpu"
    wrapper.tokenizer = MagicMock()
    wrapper.tokenizer.convert_tokens_to_ids.side_effect = {
        "<AMR>": 10,
        "</AMR>": 11,
    }.__getitem__
    wrapper.tokenizer.unk_token_id = 0
    wrapper.tokenizer.mask_token_id = 12
    wrapper.tokenizer.return_value.input_ids = torch.tensor([[1, 21, 2]])
    wrapper.tokenizer.decode.return_value = RAW_GRAPH
    wrapper.model = MagicMock()
    wrapper.model.generate.return_value = torch.tensor([[10, 20, 11]])
    return wrapper


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("consist-of", ("z0", ":consist-of", "z1")),
        ("prep-out-of", ("z0", ":prep-out-of", "z1")),
        ("prep-on-behalf-of", ("z0", ":prep-on-behalf-of", "z1")),
        ("ARG0-of", ("z1", ":ARG0", "z0")),
        ("domain", ("z0", ":domain", "z1")),
    ],
)
def test_api_triples_match_amr_aware_penman(parser, role, expected):
    parser.tokenizer.decode.return_value = (
        f"<AMR> (<pointer:0> thing :{role} (<pointer:1> person)) </AMR>"
    )
    result = parser.parse("text")

    assert result.decode_status == "ok"
    assert AMRTriple(*expected, "node") in result.triples
    # This is also how Tendora reads the returned PENMAN field.
    graph = penman.decode(result.penman, model=amr_model)
    assert expected in graph.triples
    assert graph.triples == [(t.source, t.relation, t.target) for t in result.triples]


@pytest.mark.parametrize(
    "graph",
    [
        "(<pointer:0> approve-01) (<pointer:1> reject-01 :ARG0",
        "(<pointer:0> approve-01) (<pointer:1> reject-01)",
        "(<pointer:0> approve-01) trailing",
        "(<pointer:0> approve-01))",
        "(<pointer:0> approve-01 :ARG0)",
        "(<pointer:0> approve-01 :ARG0 :ARG1 (<pointer:1> person))",
        "(<pointer:0> approve-01 :ARG0",
        "(<pointer:0> approve-01 :ARG0 (<pointer:1>))",
        "(<pointer:0> approve-01 :ARG0 (<pointer:1> <pointer:2>))",
        "(<pointer:0>)",
        "(<pointer:0> /)",
        "(<pointer:0> approve-01 :ARG0 ())",
        '(<pointer:0> string-entity :value "unclosed)',
    ],
)
def test_incomplete_or_trailing_graphs_are_retained_as_invalid(parser, graph):
    raw = f"<AMR> {graph} </AMR>"
    parser.tokenizer.decode.return_value = raw

    with pytest.raises(AMRDecodeError):
        pointer_graph_to_penman(raw)
    result = parser.parse("text")

    assert result.decode_status == "invalid"
    assert result.linearized == result.linearized_raw == raw
    assert result.penman is None
    assert result.top is None
    assert result.triples == ()
    assert result.decode_error


def test_valid_unknowns_forward_references_and_literal_syntax_are_preserved():
    raw = (
        '<AMR> (<pointer:0> ask-01 :ARG1 <pointer:1> '
        ':ARG2 (<pointer:1> amr-unknown) :polarity - :quant 2 '
        ':value <lit>(<pointer:9> thing :ARG0 x)</lit> '
        ':comment "<lit> is a marker") </AMR>'
    )
    graph = penman.decode(pointer_graph_to_penman(raw), model=amr_model)

    assert ("z1", ":instance", "amr-unknown") in graph.triples
    assert ("z0", ":ARG1", "z1") in graph.triples
    assert ("z0", ":polarity", "-") in graph.triples
    assert ("z0", ":quant", "2") in graph.triples
    assert ("z0", ":value", '"(<pointer:9> thing :ARG0 x)"') in graph.triples
    assert ("z0", ":comment", '"<lit> is a marker"') in graph.triples


@pytest.mark.parametrize("ending", ["</lit>) </AMR>", ""])
def test_invalid_fragment_scanner_does_not_interpret_literal_contents(ending):
    literal = "(<pointer:9> fake :ARG0 (<pointer:8> actor))"
    raw = f"<AMR> (<pointer:0> string-entity :value <lit>{literal}{ending}"
    fragments = scan_invalid_amr_fragments(raw)

    assert [(f.source, f.relation) for f in fragments] == [
        ("z0", ":instance"), ("z0", ":value")
    ]
    assert fragments[1].target == literal
    assert fragments[1].complete == bool(ending)
    assert raw[fragments[1].token_start:fragments[1].token_end].startswith(":value")


@pytest.mark.parametrize("text_tokens", [397, 398])
def test_source_limit_preserves_every_token_or_skips_inference(parser, text_tokens):
    source_ids = [1, *([21] * (text_tokens - 2)), 2]
    parser.tokenizer.return_value.input_ids = torch.tensor([source_ids])
    result = parser.parse("The engineer did not approve deployment.")

    assert parser.tokenizer.call_args.kwargs["truncation"] is False
    assert result.input_tokens == text_tokens + 3
    if text_tokens == 397:
        assert result.decode_status == "ok"
        model_input = parser.model.generate.call_args.kwargs["input_ids"].tolist()[0]
        assert model_input == [*source_ids, 10, 12, 11]
    else:
        parser.model.generate.assert_not_called()
        assert result.decode_status == "input_too_long"
        assert result.output_tokens == 0
        assert result.penman is None
        assert result.linearized_raw == ""
        assert "401" in result.decode_error and "400" in result.decode_error


def test_batch_keeps_valid_invalid_and_oversized_inputs_in_order(parser, monkeypatch):
    parser.tokenizer.side_effect = [
        MagicMock(input_ids=torch.tensor([[1, 21, 2]])),
        MagicMock(input_ids=torch.tensor([[1, *([21] * 398), 2]])),
        MagicMock(input_ids=torch.tensor([[1, 21, 2]])),
    ]
    invalid = "<AMR> (<pointer:0> approve-01 :ARG0) </AMR>"
    parser.tokenizer.decode.side_effect = [RAW_GRAPH, invalid]
    monkeypatch.setattr(amr, "amrbart_model_wrapper", parser)
    local_app = FastAPI()
    local_app.include_router(amr.router)

    with TestClient(local_app) as client:
        response = client.post(
            "/amr/parse-batch", json={"texts": ["valid", "oversized", "invalid"]}
        )

    assert response.status_code == 200
    results = response.json()
    assert [r["decode_status"] for r in results] == ["ok", "input_too_long", "invalid"]
    assert results[0]["penman"]
    assert results[1]["input_tokens"] == 403
    assert results[1]["output_tokens"] == 0
    assert results[2]["linearized_raw"] == invalid
    assert parser.model.generate.call_count == 2


def test_server_suppresses_penman_graph_dumps():
    assert logging.getLogger("penman").getEffectiveLevel() >= logging.WARNING


def test_pointer_graph_decoder_preserves_reentrancy():
    graph = penman.decode(pointer_graph_to_penman(RAW_GRAPH))

    assert graph.top == "z0"
    assert ("z0", ":instance", "contrast-01") in graph.triples
    assert ("z1", ":ARG1", "z2") in graph.triples
    assert ("z3", ":ARG0", "z2") in graph.triples


def test_pointer_graph_decoder_joins_split_roles_without_changing_literals():
    raw = ('<AMR> (<pointer:0> and : op6 (<pointer:1> person '
           ': prep-as (<pointer:2> artist)) : op7 <pointer:1> '
           ':value <lit>: op6 and : prep-as</lit>) </AMR>')
    graph = penman.decode(pointer_graph_to_penman(raw))
    assert ("z0", ":op6", "z1") in graph.triples
    assert ("z0", ":op7", "z1") in graph.triples
    assert ("z1", ":prep-as", "z2") in graph.triples
    assert ("z0", ":value", '\": op6 and : prep-as\"') in graph.triples
    assert ': op6' in raw


def test_split_role_normalization_does_not_supply_missing_concepts():
    with pytest.raises(AMRDecodeError, match="no concept"):
        pointer_graph_to_penman(
            '<AMR> (<pointer:0> and : op6 (<pointer:1> :ARG0 x)) </AMR>'
        )


def test_pointer_graph_decoder_lowers_checkpoint_literal_tokens():
    graph = penman.decode(
        pointer_graph_to_penman(
            "<AMR> (<pointer:0> string-entity :value <lit>UTF-8 bytes</lit>) </AMR>"
        )
    )

    assert ("z0", ":value", '"UTF-8 bytes"') in graph.triples


def test_pointer_graph_decoder_rejects_unterminated_checkpoint_literal():
    with pytest.raises(AMRDecodeError, match="unterminated literal"):
        pointer_graph_to_penman(
            "<AMR> (<pointer:0> string-entity :value <lit>UTF-8) </AMR>"
        )


def test_invalid_fragment_scanner_retains_diagnostic_relations_only():
    fragments = scan_invalid_amr_fragments(
        "<AMR> (<pointer:0> change-01 :ARG0 <pointer:9> :ARG1 <lit>bad literal) </AMR>"
    )

    assert fragments[0] == AMRFragment(
        source="z0",
        relation=":instance",
        target="change-01",
        target_kind="literal",
        token_start=7,
        token_end=28,
        complete=True,
    )
    assert fragments[1].target == "z9"
    assert fragments[1].complete
    assert fragments[2].target == "bad literal) </AMR>"
    assert not fragments[2].complete


def test_model_returns_ordered_validated_triples(monkeypatch):
    wrapper = AMRBartModelWrapper()
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.side_effect = lambda token: {
        "<AMR>": 10,
        "</AMR>": 11,
    }[token]
    tokenizer.unk_token_id = 0
    tokenizer.mask_token_id = 12
    tokenizer.return_value.input_ids.__getitem__.return_value.tolist.return_value = [
        1,
        2,
    ]
    tokenizer.decode.return_value = RAW_GRAPH
    generated = MagicMock()
    generated_ids = generated.__getitem__.return_value.detach.return_value.cpu
    generated_ids.return_value.tolist.return_value = [10, 20, 11]
    model = MagicMock()
    model.generate.return_value = generated
    wrapper.tokenizer = tokenizer
    wrapper.model = model
    wrapper.device = "cpu"

    monkeypatch.setattr("src.models.amrbart.torch.tensor", MagicMock())
    monkeypatch.setattr("src.models.amrbart.torch.ones_like", MagicMock())
    result = wrapper.parse("text")

    assert result.top == "z0"
    assert result.triples[0] == AMRTriple("z0", ":instance", "contrast-01", "literal")
    assert AMRTriple("z1", ":ARG1", "z2", "node") in result.triples


def test_pointer_graph_decoder_rejects_unknown_reference():
    with pytest.raises(AMRDecodeError, match="undeclared pointer.*9"):
        pointer_graph_to_penman(
            "<AMR> (<pointer:0> change-01 :ARG0 <pointer:9>) </AMR>"
        )


def test_pointer_graph_decoder_rejects_duplicate_declaration():
    with pytest.raises(AMRDecodeError, match="declared more than once"):
        pointer_graph_to_penman(
            "<AMR> (<pointer:0> and :op1 (<pointer:1> one) "
            ":op2 (<pointer:1> two)) </AMR>"
        )


def test_model_parse_retains_invalid_raw_output(monkeypatch):
    wrapper = AMRBartModelWrapper()
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.side_effect = lambda token: {
        "<AMR>": 10,
        "</AMR>": 11,
    }[token]
    tokenizer.unk_token_id = 0
    tokenizer.mask_token_id = 12
    tokenizer.return_value.input_ids = MagicMock()
    tokenizer.return_value.input_ids.__getitem__.return_value.tolist.return_value = [
        1,
        2,
    ]
    tokenizer.decode.return_value = (
        "<AMR> (<pointer:0> change-01 :ARG0 <pointer:9>) </AMR>"
    )

    model = MagicMock()
    generated = MagicMock()
    output_ids = generated.__getitem__.return_value.detach.return_value.cpu
    output_ids.return_value.tolist.return_value = [10, 20, 11]
    model.generate.return_value = generated
    wrapper.tokenizer = tokenizer
    wrapper.model = model
    wrapper.device = "cpu"

    monkeypatch.setattr("src.models.amrbart.torch.tensor", MagicMock())
    monkeypatch.setattr("src.models.amrbart.torch.ones_like", MagicMock())
    result = wrapper.parse("text")

    assert result.decode_status == "invalid"
    assert result.penman is None
    assert result.linearized.startswith("<AMR>")
    assert "undeclared pointer" in result.decode_error
    assert result.fragments


def test_amr_endpoint_returns_attributed_candidate():
    from src.config import settings

    result = AMRParseResult(
        model="parser",
        revision="revision",
        device="cpu",
        input_tokens=12,
        output_tokens=24,
        linearized=RAW_GRAPH,
        penman=pointer_graph_to_penman(RAW_GRAPH),
        top="z0",
        triples=(AMRTriple("z0", ":instance", "contrast-01", "literal"),),
        fragments=(),
        decode_status="ok",
    )
    wrapper = MagicMock()
    wrapper.is_loaded = True
    wrapper.parse.return_value = result
    app.dependency_overrides[get_api_key] = lambda: "test-key"

    try:
        with (
            patch("src.server.SentenceTransformerWrapper") as embedding,
            patch("src.server.SummarizationModelWrapper") as summary,
            patch("src.server.ConversationManager"),
            patch.object(settings, "MODEL_PROFILE", "workbench"),
            patch.object(settings, "CHAT_MODEL_ENABLED", False),
            patch.object(settings, "AMRBART_MODEL_ENABLED", False),
        ):
            with TestClient(app, raise_server_exceptions=True) as client:
                embedding.return_value.model = MagicMock()
                summary.return_value.model = MagicMock()
                amr.amrbart_model_wrapper = wrapper
                response = client.post("/amr/parse", json={"text": "test"})
    finally:
        app.dependency_overrides.pop(get_api_key, None)
        amr.amrbart_model_wrapper = None

    assert response.status_code == 200
    assert response.json()["decode_status"] == "ok"
    assert response.json()["revision"] == "revision"
    assert response.json()["triples"] == [
        {
            "source": "z0",
            "relation": ":instance",
            "target": "contrast-01",
            "target_kind": "literal",
        }
    ]
    assert response.json()["fragments"] == []
    wrapper.parse.assert_called_once_with("test")


def test_amr_batch_endpoint_retains_each_attempt():
    result = AMRParseResult(
        model="parser",
        revision="revision",
        device="cpu",
        input_tokens=12,
        output_tokens=24,
        linearized=RAW_GRAPH,
        penman=pointer_graph_to_penman(RAW_GRAPH),
        top="z0",
        triples=(AMRTriple("z0", ":instance", "contrast-01", "literal"),),
        fragments=(),
        decode_status="ok",
    )
    wrapper = MagicMock()
    wrapper.is_loaded = True
    wrapper.parse.return_value = result
    app.dependency_overrides[get_api_key] = lambda: "test-key"

    try:
        with (
            patch("src.server.SentenceTransformerWrapper") as embedding,
            patch("src.server.SummarizationModelWrapper"),
            patch("src.server.ConversationManager"),
            patch.object(settings, "MODEL_PROFILE", "workbench"),
            patch.object(settings, "CHAT_MODEL_ENABLED", False),
            patch.object(settings, "AMRBART_MODEL_ENABLED", False),
        ):
            embedding.return_value.model = MagicMock()
            with TestClient(app) as client:
                amr.amrbart_model_wrapper = wrapper
                response = client.post(
                    "/amr/parse-batch", json={"texts": ["one", "two"]}
                )
    finally:
        app.dependency_overrides.pop(get_api_key, None)
        amr.amrbart_model_wrapper = None

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert wrapper.parse.call_count == 2


def test_amr_endpoint_requires_loaded_model():
    amr.amrbart_model_wrapper = None
    app.dependency_overrides[get_api_key] = lambda: "test-key"
    try:
        client = TestClient(app)
        response = client.post("/amr/parse", json={"text": "test"})
    finally:
        app.dependency_overrides.pop(get_api_key, None)

    assert response.status_code == 503


def test_text_embedding_batch_returns_pinned_attributed_vectors():
    from src.config import settings

    app.dependency_overrides[get_api_key] = lambda: "test-key"
    try:
        with (
            patch("src.server.SentenceTransformerWrapper") as embedding,
            patch("src.server.SummarizationModelWrapper"),
            patch("src.server.ConversationManager"),
            patch.object(settings, "MODEL_PROFILE", "workbench"),
            patch.object(settings, "CHAT_MODEL_ENABLED", False),
            patch.object(settings, "AMRBART_MODEL_ENABLED", False),
        ):
            embedding.return_value.model = MagicMock()
            embedding.return_value.encode_many.return_value = np.array(
                [np.arange(128, dtype=float), np.arange(128, dtype=float) + 128]
            )
            with TestClient(app) as client:
                response = client.post(
                    "/embed/texts",
                    json={
                        "texts": ["one", "two"],
                        "dimensions": 128,
                        "prompt_name": "STS",
                    },
                )
    finally:
        app.dependency_overrides.pop(get_api_key, None)
        embed.embedding_model_wrapper = None

    assert response.status_code == 200
    assert response.json()["model"] == settings.EMBEDDING_MODEL_NAME
    assert response.json()["revision"] == settings.EMBEDDING_MODEL_REVISION
    assert response.json()["prompt_name"] == "STS"
    assert len(response.json()["embeddings"]) == 2
    assert all(len(vector) == 128 for vector in response.json()["embeddings"])


def test_semantic_profile_selects_only_semantic_models(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "MODEL_PROFILE", "semantic")
    monkeypatch.setattr(settings, "CHAT_MODEL_ENABLED", True)
    monkeypatch.setattr(settings, "SUMMARY_LOCAL_ENABLED", True)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "configured")

    assert settings.amrbart_enabled
    assert not settings.chat_enabled
    assert not settings.local_summary_enabled
    assert not settings.gemini_summary_enabled


def test_semantic_profile_lifespan_loads_embedding_and_amrbart_only():
    from src.config import settings

    with (
        patch.object(settings, "MODEL_PROFILE", "semantic"),
        patch.object(settings, "GEMINI_API_KEY", "configured"),
        patch("src.server.SentenceTransformerWrapper") as embedding,
        patch("src.server.AMRBartModelWrapper") as parser,
        patch("src.server.SummarizationModelWrapper") as summary,
        patch("src.server.ChatModelWrapper") as chat,
        patch("src.server.ConversationManager") as conversation_manager,
    ):
        embedding.return_value.model = MagicMock()
        parser.return_value.is_loaded = True
        with TestClient(app):
            pass

    embedding.return_value.load_model.assert_called_once_with()
    parser.return_value.load_model.assert_called_once_with()
    summary.return_value.load_local_model.assert_not_called()
    summary.return_value.load_gemini_client.assert_not_called()
    chat.assert_not_called()
    conversation_manager.assert_not_called()
