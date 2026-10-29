import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.models.corpus_gpt import Cancelled, train_corpus_gpt
from src.routers import corpus

SENTENCES = [
    ["a", "keeper", "holds", "the", "key", "."],
    ["the", "keeper", "rotates", "the", "key", "."],
    ["a", "guest", "holds", "no", "key", "."],
]
WORDS = {"keeper-01": "keeper", "key": "key", "absent": "door"}
POSITIONS = {"keeper-01": [[0, 1], [1, 1]], "key": [[0, 4], [1, 4], [2, 4]]}


def small(**more):
    return train_corpus_gpt(
        SENTENCES,
        WORDS,
        POSITIONS,
        steps=30,
        dimensions=16,
        layers=2,
        heads=2,
        threads=1,
        **more,
    )


def test_reads_back_each_key_at_its_word_and_its_places():
    out = small()
    assert set(out["wte"]) == {"keeper-01", "key"}  # "door" is in no sentence
    assert all(len(v) == 16 for v in out["wte"].values())
    assert len(out["layers"]) == 2
    assert set(out["layers"][0]) == {"keeper-01", "key"}
    assert out["model"]["tokens"] == 18 and out["model"]["vocab"] == 3 + 9


def test_the_same_seed_gives_the_same_network():
    assert small()["layers"][1]["key"] == small()["layers"][1]["key"]
    assert small(seed=8)["wte"]["key"] != small()["wte"]["key"]


def test_reports_progress_and_stops_when_cancelled():
    seen = []
    small(progress=lambda step, steps, loss: seen.append(step), every=10)
    assert seen == [10, 20, 30]
    with pytest.raises(Cancelled):
        small(cancelled=lambda: True)


def test_refuses_a_place_outside_the_sentences():
    with pytest.raises(ValueError):
        train_corpus_gpt(SENTENCES, positions={"x": [[0, 6]]}, steps=1)


def test_streams_progress_then_the_result():
    app = FastAPI()
    app.include_router(corpus.router)
    body = {
        "sentences": SENTENCES,
        "words": WORDS,
        "positions": POSITIONS,
        "steps": 50,
        "dimensions": 16,
        "layers": 2,
        "heads": 2,
        "threads": 1,
    }
    with TestClient(app).stream("POST", "/corpus/gpt", json=body) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    events = [
        (
            block.split("\n")[0].removeprefix("event: "),
            json.loads(block.split("\n")[1][6:]),
        )
        for block in text.strip().split("\n\n")
    ]
    assert [name for name, _ in events] == ["progress", "progress", "result"]
    assert events[1][1]["step"] == 50 and events[-1][1]["model"]["steps"] == 50
    one = TestClient(app).post("/corpus/gpt", json={**body, "stream": False}).json()
    assert set(one["wte"]) == {"keeper-01", "key"}


@pytest.mark.parametrize(
    ("sentences", "detail"),
    [
        ([["w"] * (corpus.MAX_SENTENCE_WORDS + 1)], "at most 256 items"),
        (
            [
                [f"w{k}" for k in range(start, start + 250)]
                for start in range(0, corpus.MAX_VOCABULARY + 1, 250)
            ],
            "distinct words",
        ),
    ],
    ids=["sentence too long", "vocabulary too large"],
)
def test_corpus_bounds_are_refused(sentences, detail):
    app = FastAPI()
    app.include_router(corpus.router)
    body = {"sentences": sentences, "stream": False}
    response = TestClient(app).post("/corpus/gpt", json=body)
    assert response.status_code == 422
    assert detail in json.dumps(response.json())
